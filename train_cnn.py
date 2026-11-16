"""
train_cnn.py
------------
End-to-End PyTorch Multi-Class CNN Face Classification Training Pipeline.

Features:
- PyTorch Dataset & DataLoader with Torchvision Transforms
- Exact preprocessing parity with face_detector.py (112x112, ImageNet normalization)
- Class imbalance handling via Weighted CrossEntropyLoss or WeightedRandomSampler
- AdamW optimizer with weight decay
- Learning rate scheduling with ReduceLROnPlateau
- Early stopping & model checkpointing saving 'models/best_model.pth'
- Detailed epoch logs and training curves visualization saved to 'static/images/training_metrics.png'
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DATASET_DIR, MODELS_DIR, CNN_MODEL_PATH, CLASS_NAMES_PATH,
    METRICS_PLOT_PATH, IMAGE_SIZE, BATCH_SIZE, LEARNING_RATE,
    WEIGHT_DECAY, DEFAULT_EPOCHS, EARLY_STOPPING_PATIENCE,
    NORM_MEAN, NORM_STD
)
from models.face_cnn import FaceCNN, build_transfer_face_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TrainCNN")


# ----------------------------------------------------------------------
# Data Transforms (Training vs Validation/Test)
# ----------------------------------------------------------------------

def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns data transforms for training and validation.
    Training uses photorealistic augmentations (slight rotation, jitter, flip).
    Validation uses deterministic resizing and identical normalization.
    """
    train_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-12, 12)),
        transforms.ColorJitter(brightness=0.18, contrast=0.18, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    return train_transform, val_transform


# ----------------------------------------------------------------------
# Class Imbalance Handler
# ----------------------------------------------------------------------

def compute_class_weights(dataset: datasets.ImageFolder) -> Tuple[torch.Tensor, Optional[WeightedRandomSampler]]:
    """
    Analyzes class distributions and computes inverse-frequency weights for CrossEntropyLoss.
    Also creates an optional WeightedRandomSampler.
    """
    targets = [s[1] for s in dataset.samples]
    class_counts = [0] * len(dataset.classes)
    for t in targets:
        class_counts[t] += 1

    total_samples = len(targets)
    num_classes = len(dataset.classes)

    logger.info("Class distribution in training set: %s (Total: %d)",
                {dataset.classes[i]: class_counts[i] for i in range(num_classes)}, total_samples)

    # Calculate inverse-frequency weights: w_c = N / (C * N_c)
    weights = [total_samples / (num_classes * max(1, count)) for count in class_counts]
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32)

    # Per-sample weights for WeightedRandomSampler
    sample_weights = [weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return class_weights_tensor, sampler


# ----------------------------------------------------------------------
# Training Curves Visualization
# ----------------------------------------------------------------------

def plot_training_metrics(history: Dict[str, List[float]], save_path: str = METRICS_PLOT_PATH):
    """Renders professional accuracy and loss curves for documentation and UI display."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        epochs = range(1, len(history["train_loss"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
        fig.patch.set_facecolor("#0F172A")

        for ax in (ax1, ax2):
            ax.set_facecolor("#1E293B")
            ax.tick_params(colors="#94A3B8")
            ax.spines["bottom"].set_color("#334155")
            ax.spines["top"].set_color("#334155")
            ax.spines["left"].set_color("#334155")
            ax.spines["right"].set_color("#334155")
            ax.grid(True, color="#334155", linestyle="--", alpha=0.5)

        # Accuracy Curve
        ax1.plot(epochs, history["train_acc"], color="#38BDF8", linewidth=2.2, label="Train Accuracy")
        ax1.plot(epochs, history["val_acc"], color="#4ADE80", linewidth=2.2, linestyle="--", label="Val Accuracy")
        ax1.set_title("CNN Multi-Class Accuracy", color="#F8FAFC", fontsize=12, fontweight="bold", pad=10)
        ax1.set_xlabel("Epoch", color="#94A3B8", fontsize=10)
        ax1.set_ylabel("Accuracy", color="#94A3B8", fontsize=10)
        ax1.set_ylim(0.0, 1.05)
        ax1.legend(loc="lower right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        # Loss Curve
        ax2.plot(epochs, history["train_loss"], color="#F43F5E", linewidth=2.2, label="Train Loss")
        ax2.plot(epochs, history["val_loss"], color="#FB923C", linewidth=2.2, linestyle="--", label="Val Loss")
        ax2.set_title("Cross-Entropy Loss", color="#F8FAFC", fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Epoch", color="#94A3B8", fontsize=10)
        ax2.set_ylabel("Loss", color="#94A3B8", fontsize=10)
        ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#F8FAFC")

        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info("Saved training curves plot to %s", save_path)
    except Exception as e:
        logger.warning("Failed to render training curves: %s", e)


# ----------------------------------------------------------------------
# Main Training Routine
# ----------------------------------------------------------------------

def train_cnn(
    dataset_dir: str = DATASET_DIR,
    model_save_path: str = CNN_MODEL_PATH,
    class_names_path: str = CLASS_NAMES_PATH,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    use_transfer_learning: bool = False,
    device_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end PyTorch training for the Multi-Class CNN:
    1. Loads dataset/train and dataset/validation
    2. Builds CNN model matching num_classes
    3. Handles class imbalance via weighted loss
    4. Trains with AdamW, ReduceLROnPlateau, EarlyStopping
    5. Saves best_model.pth, class_names.json, and metrics plot
    """
    # Select compute device (MPS on Apple Silicon, CUDA on NVIDIA, or CPU)
    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s for training", device)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "validation")

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Dataset directories missing. Run dataset_builder.py first. (Checked {train_dir})")

    train_tf, val_tf = get_data_transforms()

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    num_classes = len(train_dataset.classes)
    if num_classes < 2:
        raise ValueError(f"Need at least 2 registered classes to train CNN. Found: {num_classes}")

    # Build Class Names Mapping
    class_names = {str(i): cname for i, cname in enumerate(train_dataset.classes)}
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    logger.info("Registered %d classes: %s", num_classes, list(class_names.values()))

    # Compute class weights for handling dataset imbalance
    class_weights, sampler = compute_class_weights(train_dataset)
    class_weights = class_weights.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize Architecture
    if use_transfer_learning:
        logger.info("Initializing Transfer Learning model (MobileNetV3-Small) with %d outputs...", num_classes)
        model = build_transfer_face_model(num_classes=num_classes, pretrained=True)
    else:
        logger.info("Initializing Custom Deep FaceCNN (4 Conv Blocks) with %d outputs...", num_classes)
        model = FaceCNN(num_classes=num_classes, dropout_rate=0.4)

    model = model.to(device)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": []
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    logger.info("Starting training loop for %d epochs...", epochs)
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # 2. Validation Phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        scheduler.step(epoch_val_acc)

        logger.info(
            "Epoch [%02d/%02d] | Train Loss: %.4f Acc: %.2f%% | Val Loss: %.4f Val Acc: %.2f%%",
            epoch, epochs, epoch_train_loss, epoch_train_acc * 100, epoch_val_loss, epoch_val_acc * 100
        )

        # Checkpointing Best Model
        if epoch_val_acc > best_val_acc or (epoch_val_acc == best_val_acc and epoch_val_loss < history["val_loss"][best_epoch - 1]):
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": best_val_acc,
                "val_loss": epoch_val_loss,
                "architecture": "MobileNetV3-Small" if use_transfer_learning else "FaceCNN_4Block",
                "norm_mean": NORM_MEAN,
                "norm_std": NORM_STD,
                "image_size": IMAGE_SIZE
            }
            torch.save(checkpoint, model_save_path)
            # Also export compact production deployment checkpoint (pure weights, 66.7% smaller)
            deploy_save_path = os.path.splitext(model_save_path)[0] + "_deploy.pth"
            deploy_ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "val_acc": best_val_acc,
                "val_loss": epoch_val_loss,
                "architecture": "MobileNetV3-Small" if use_transfer_learning else "FaceCNN_4Block",
                "norm_mean": NORM_MEAN,
                "norm_std": NORM_STD,
                "image_size": IMAGE_SIZE
            }
            torch.save(deploy_ckpt, deploy_save_path)
            logger.info("⭐ New best model saved to %s (Val Acc: %.2f%%) | Deploy: %s",
                        model_save_path, best_val_acc * 100, deploy_save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch)
                break

    elapsed = time.time() - start_time
    logger.info("Training completed in %.1f seconds. Best Validation Accuracy: %.2f%% (Epoch %d)",
                elapsed, best_val_acc * 100, best_epoch)

    # Plot metrics
    plot_training_metrics(history, save_path=METRICS_PLOT_PATH)

    # Automatic open-world threshold calibration
    try:
        from calibrate_threshold import calibrate_threshold
        calibrate_threshold(model_path=model_save_path)
    except Exception as cal_err:
        logger.warning("Could not automatically run threshold calibration: %s", cal_err)

    summary = {
        "success": True,
        "epochs_run": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_acc, 4),
        "final_train_accuracy": round(history["train_acc"][-1], 4),
        "model_path": model_save_path,
        "class_names_path": class_names_path,
        "num_classes": num_classes,
        "classes": class_names
    }

    # Save training summary
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Class Face CNN Classifier")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--transfer", action="store_true", help="Use MobileNetV3 Transfer Learning")
    args = parser.parse_args()

    print("=" * 65)
    print("Multi-Class Face CNN Classifier Training Pipeline (PyTorch)")
    print("=" * 65)

    res = train_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_transfer_learning=args.transfer
    )

    print("\n--- Training Result Summary ---")
    print(json.dumps(res, indent=2))
