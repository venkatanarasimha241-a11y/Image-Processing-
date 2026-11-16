"""
evaluate_model.py
-----------------
Comprehensive Multi-Class CNN Face Recognition Evaluation.

Calculates:
- Overall Test Accuracy
- Per-Class Precision, Recall, and F1-Score
- Multi-Class Confusion Matrix (Heatmap visualization saved to 'static/images/confusion_matrix.png')
- Open-World Unknown Rejection & False Acceptance Rate on 'unknown_test/'
- Misclassification and Confusion Pair Analysis
"""

import os
import json
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# Optional sklearn metrics with pure NumPy fallback
try:
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support  # type: ignore
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    classification_report = None  # type: ignore
    confusion_matrix = None  # type: ignore
    precision_recall_fscore_support = None  # type: ignore

def compute_metrics_numpy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure NumPy computation of Confusion Matrix, Precision, Recall, F1, and Support."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precision = np.zeros(num_classes, dtype=float)
    recall = np.zeros(num_classes, dtype=float)
    f1 = np.zeros(num_classes, dtype=float)
    support = cm.sum(axis=1)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        score = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

        precision[i] = p
        recall[i] = r
        f1[i] = score

    return cm, precision, recall, f1, support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional

from config import (
    DATASET_DIR, UNKNOWN_TEST_DIR, CNN_MODEL_PATH,
    RECOGNITION_CONFIG_PATH, CONFUSION_MATRIX_PATH,
    IMAGE_SIZE, NORM_MEAN, NORM_STD
)
from calibrate_threshold import load_model_for_inference, extract_embeddings_and_labels

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("EvaluateModel")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str = CONFUSION_MATRIX_PATH):
    """Generates an aesthetic dark-themed Confusion Matrix Heatmap."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        fig.patch.set_facecolor("#0F172A")
        ax.set_facecolor("#1E293B")

        cax = ax.matshow(cm, cmap="Blues", alpha=0.88)
        fig.colorbar(cax, fraction=0.046, pad=0.04)

        num_classes = len(class_names)
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels(class_names, rotation=35, ha="left", color="#F8FAFC", fontsize=9)
        ax.set_yticklabels(class_names, color="#F8FAFC", fontsize=9)

        ax.tick_params(colors="#94A3B8")
        ax.spines["bottom"].set_color("#334155")
        ax.spines["top"].set_color("#334155")
        ax.spines["left"].set_color("#334155")
        ax.spines["right"].set_color("#334155")

        # Overlay counts inside cells
        for i in range(num_classes):
            for j in range(num_classes):
                val = cm[i, j]
                color = "#FFFFFF" if val > cm.max() / 2 else "#94A3B8"
                ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=11, fontweight="bold")

        ax.set_xlabel("Predicted Identity", color="#38BDF8", fontsize=11, fontweight="bold", labelpad=12)
        ax.set_ylabel("True Ground Truth", color="#38BDF8", fontsize=11, fontweight="bold", labelpad=12)
        ax.set_title("Multi-Class Face Recognition Confusion Matrix", color="#F8FAFC", fontsize=13, fontweight="bold", pad=20)

        plt.tight_layout()
        plt.savefig(save_path, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        logger.info("Saved Confusion Matrix Heatmap to %s", save_path)
    except Exception as e:
        logger.warning("Failed to render confusion matrix: %s", e)


def evaluate_test_set(
    dataset_dir: str = DATASET_DIR,
    unknown_dir: str = UNKNOWN_TEST_DIR,
    model_path: str = CNN_MODEL_PATH,
    config_path: str = RECOGNITION_CONFIG_PATH
) -> Dict[str, Any]:
    """
    Evaluates trained CNN on hold-out Test split and unseen Unknown dataset.
    Generates precision, recall, F1, and confusion matrix.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    logger.info("Evaluating CNN Model on %s...", device)

    model, class_names_dict = load_model_for_inference(model_path, device=device)
    class_names = [class_names_dict[str(i)] for i in range(len(class_names_dict))]

    # Load calibrated threshold if available
    threshold = 0.80
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cdata = json.load(f)
                threshold = float(cdata.get("calibrated_threshold", 0.80))
        except Exception:
            pass

    test_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    test_dir = os.path.join(dataset_dir, "test")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test split not found at {test_dir}")

    test_dataset = datasets.ImageFolder(test_dir, transform=test_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true: List[int] = []
    y_pred: List[int] = []
    confidences: List[float] = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=-1)
            top_conf, preds = torch.max(probs, dim=-1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            confidences.extend(top_conf.cpu().numpy().tolist())

    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    conf_np = np.array(confidences)

    # 1. Overall & Per-Class Metrics
    acc = float(np.mean(y_true_np == y_pred_np))
    prec_arr: np.ndarray
    rec_arr: np.ndarray
    f1_arr: np.ndarray
    supp_arr: np.ndarray
    if HAS_SKLEARN and callable(precision_recall_fscore_support) and callable(confusion_matrix):
        res_metrics: Any = precision_recall_fscore_support(  # type: ignore
            y_true_np, y_pred_np, labels=list(range(len(class_names))), zero_division=0.0  # type: ignore
        )
        prec_arr = np.asarray(res_metrics[0])
        rec_arr = np.asarray(res_metrics[1])
        f1_arr = np.asarray(res_metrics[2])
        supp_arr = np.asarray(res_metrics[3]) if res_metrics[3] is not None else np.zeros(len(class_names), dtype=int)
        cm = confusion_matrix(y_true_np, y_pred_np, labels=list(range(len(class_names))))  # type: ignore
    else:
        cm, prec_arr, rec_arr, f1_arr, supp_arr = compute_metrics_numpy(y_true_np, y_pred_np, len(class_names))

    per_class_metrics: Dict[str, Dict[str, float]] = {}
    for i, cname in enumerate(class_names):
        per_class_metrics[cname] = {
            "precision": round(float(prec_arr[i]), 4),
            "recall": round(float(rec_arr[i]), 4),
            "f1_score": round(float(f1_arr[i]), 4),
            "support": int(supp_arr[i])
        }

    # 2. Confusion Matrix
    plot_confusion_matrix(cm, class_names)

    # Identify confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "actual": class_names[i],
                    "predicted": class_names[j],
                    "count": int(cm[i, j])
                })

    # 3. Unknown Impostor Test (Open-World Evaluation)
    unknown_results = {
        "total_unknown_evaluated": 0,
        "false_acceptance_count": 0,
        "false_acceptance_rate": 0.0,
        "unknown_rejection_rate": 1.0
    }

    if os.path.exists(unknown_dir) and len(os.listdir(unknown_dir)) > 0:
        val_dataset = datasets.ImageFolder(os.path.join(dataset_dir, "validation"), transform=test_tf)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        val_embs, val_lbls = extract_embeddings_and_labels(model, val_loader, device)

        gallery_centroids = {}
        for c in np.unique(val_lbls):
            cnt = val_embs[val_lbls == c].mean(axis=0)
            norm = np.linalg.norm(cnt)
            gallery_centroids[c] = cnt / max(1e-6, norm)

        unknown_dataset = datasets.ImageFolder(unknown_dir, transform=test_tf)
        unknown_loader = DataLoader(unknown_dataset, batch_size=16, shuffle=False)
        unk_embs, _ = extract_embeddings_and_labels(model, unknown_loader, device)

        u_confs: List[float] = []
        for emb in unk_embs:
            sims = [float(np.dot(emb, cnt)) for cnt in gallery_centroids.values()]
            u_confs.append(max(sims))

        total_u = len(u_confs)
        fa_count = sum(1 for c in u_confs if c >= threshold)
        far = fa_count / max(1, total_u)
        unknown_results = {
            "total_unknown_evaluated": total_u,
            "false_acceptance_count": fa_count,
            "false_acceptance_rate": round(far, 4),
            "unknown_rejection_rate": round(1.0 - far, 4),
            "evaluation_threshold": round(threshold, 3),
            "impostor_similarity_min": round(float(min(u_confs)), 4) if u_confs else 0.0,
            "impostor_similarity_mean": round(float(np.mean(u_confs)), 4) if u_confs else 0.0,
            "impostor_similarity_max": round(float(max(u_confs)), 4) if u_confs else 0.0
        }

    report = {
        "model_path": model_path,
        "overall_test_accuracy": round(acc, 4),
        "calibrated_threshold_used": round(threshold, 3),
        "total_test_samples": len(y_true),
        "num_classes": len(class_names),
        "per_class_metrics": per_class_metrics,
        "confused_pairs": confused_pairs,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_plot": CONFUSION_MATRIX_PATH,
        "unknown_impostor_evaluation": unknown_results
    }

    # Save evaluation report
    report_path = os.path.join(os.path.dirname(model_path), "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved complete evaluation report to %s", report_path)

    return report


if __name__ == "__main__":
    print("=" * 65)
    print("Multi-Class CNN Face Model Comprehensive Evaluation")
    print("=" * 65)

    rep = evaluate_test_set()
    print("\n--- Evaluation Summary ---")
    print(f"Overall Test Accuracy: {rep['overall_test_accuracy'] * 100:.2f}%")
    print(f"Threshold Used:        {rep['calibrated_threshold_used']:.3f}")
    print("\nPer-Class Performance:")
    for cname, m in rep["per_class_metrics"].items():
        print(f"  - {cname:15s}: Precision={m['precision']:.2f}, Recall={m['recall']:.2f}, F1={m['f1_score']:.2f}")

    print("\nUnknown Rejection Test:")
    u = rep["unknown_impostor_evaluation"]
    print(f"  - Unknown Samples:        {u['total_unknown_evaluated']}")
    print(f"  - Unknown Rejection Rate: {u['unknown_rejection_rate'] * 100:.2f}%")
    print(f"  - False Acceptance Rate:  {u['false_acceptance_rate'] * 100:.2f}%")
    print("\nConfusion Matrix Heatmap saved to: static/images/confusion_matrix.png")
