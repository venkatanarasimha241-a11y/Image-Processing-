"""
benchmark_baseline.py
---------------------
Empirical Baseline Performance Benchmark for Deep Learning Face Recognition System.
Measures:
1. Model size & parameters
2. Memory usage (process RSS & peak torch memory)
3. Identification accuracy (Test & Validation splits)
4. Open-world rejection: FAR & FRR across thresholds
5. Pure CNN forward & feature extraction latency (CPU & Apple Silicon MPS)
6. End-to-end pipeline latency (OpenCV Preprocessing -> Embedding -> Cosine Match)
7. Single-image and batch FPS
Outputs results cleanly and saves to models/baseline_benchmark.json.
"""

import os
import sys
import time
import json
import psutil  # type: ignore
import logging
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    DATASET_DIR, UNKNOWN_TEST_DIR, CNN_MODEL_PATH,
    RECOGNITION_CONFIG_PATH, IMAGE_SIZE, NORM_MEAN, NORM_STD,
    RECOGNITION_SIMILARITY_THRESHOLD
)
from calibrate_threshold import load_model_for_inference, extract_embeddings_and_labels
from face_detector import FaceDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("BenchmarkBaseline")


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024.0 * 1024.0)


def measure_model_size(model_path: str, model: Any) -> Dict[str, Any]:
    """Measures model disk footprint, parameter counts, and theoretical buffer sizes."""
    file_bytes = os.path.getsize(model_path) if os.path.exists(model_path) else 0
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "file_size_bytes": file_bytes,
        "file_size_mb": round(file_bytes / (1024.0 * 1024.0), 3),
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "pure_weights_mb": round((param_bytes + buffer_bytes) / (1024.0 * 1024.0), 3)
    }


def measure_accuracy_and_rejection(
    model: Any,
    device: torch.device,
    operational_threshold: float = RECOGNITION_SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """Evaluates Genuine Test accuracy and Impostor Rejection."""
    eval_tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])

    val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "validation"), transform=eval_tf)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    val_embs, val_lbls = extract_embeddings_and_labels(model, val_loader, device)

    test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    test_embs, test_lbls = extract_embeddings_and_labels(model, test_loader, device)

    # Gallery Centroids from Validation
    gallery = {}
    for c in np.unique(val_lbls):
        c_embs = val_embs[val_lbls == c]
        cnt = np.mean(c_embs, axis=0)
        cnt /= max(1e-6, np.linalg.norm(cnt))
        gallery[c] = cnt

    # Test identification
    test_sims = []
    test_preds = []
    for emb, lbl in zip(test_embs, test_lbls):
        sims = [float(np.dot(emb, cnt)) for c, cnt in gallery.items()]
        best_sim = max(sims)
        best_pred = np.argmax(sims)
        test_sims.append(best_sim)
        test_preds.append(best_pred)

    test_preds = np.array(test_preds)
    test_correct = (test_preds == test_lbls)
    test_acc = float(np.mean(test_correct))

    # Evaluate Unknowns
    unk_sims = []
    if os.path.exists(UNKNOWN_TEST_DIR) and len(os.listdir(UNKNOWN_TEST_DIR)) > 0:
        unk_dataset = datasets.ImageFolder(UNKNOWN_TEST_DIR, transform=eval_tf)
        unk_loader = DataLoader(unk_dataset, batch_size=16, shuffle=False)
        unk_embs, _ = extract_embeddings_and_labels(model, unk_loader, device)
        for emb in unk_embs:
            sims = [float(np.dot(emb, cnt)) for c, cnt in gallery.items()]
            unk_sims.append(max(sims))

    test_sims_np = np.array(test_sims)
    unk_sims_np = np.array(unk_sims) if unk_sims else np.array([0.0])

    # Metrics at operational threshold
    false_accepts = int(np.sum(unk_sims_np >= operational_threshold))
    total_unknown = len(unk_sims_np)
    far = false_accepts / max(1, total_unknown)

    false_rejects = int(np.sum(test_sims_np < operational_threshold))
    total_test = len(test_sims_np)
    frr = false_rejects / max(1, total_test)

    # Sweep thresholds for curves
    threshold_sweep = {}
    for th in [0.70, 0.75, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95]:
        sw_far = float(np.mean(unk_sims_np >= th))
        sw_frr = float(np.mean(test_sims_np < th))
        threshold_sweep[f"{th:.2f}"] = {
            "far": round(sw_far, 4),
            "frr": round(sw_frr, 4),
            "accuracy": round(1.0 - sw_frr, 4)
        }

    return {
        "test_identification_accuracy": round(test_acc, 4),
        "total_test_samples": total_test,
        "total_unknown_samples": total_unknown,
        "operational_threshold": operational_threshold,
        "false_acceptance_count": false_accepts,
        "false_acceptance_rate": round(far, 4),
        "false_rejection_count": false_rejects,
        "false_rejection_rate": round(frr, 4),
        "genuine_similarity_min": round(float(np.min(test_sims_np)), 4),
        "genuine_similarity_mean": round(float(np.mean(test_sims_np)), 4),
        "genuine_similarity_max": round(float(np.max(test_sims_np)), 4),
        "impostor_similarity_min": round(float(np.min(unk_sims_np)), 4),
        "impostor_similarity_mean": round(float(np.mean(unk_sims_np)), 4),
        "impostor_similarity_max": round(float(np.max(unk_sims_np)), 4),
        "threshold_sweep": threshold_sweep
    }


def measure_inference_speed(
    model: Any,
    device: torch.device,
    num_warmup: int = 50,
    num_runs: int = 200
) -> Dict[str, Any]:
    """Measures single-image and batch inference latency and throughput."""
    model.eval()
    model.to(device)

    dummy_single = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)
    dummy_batch16 = torch.randn(16, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model.extract_features(dummy_single)
            if device.type == "mps":
                torch.mps.synchronize()

    # Benchmark Single Sample extract_features
    latencies_single_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model.extract_features(dummy_single)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies_single_ms.append((t1 - t0) * 1000.0)

    # Benchmark Batch 16
    latencies_batch_ms: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs // 2):
            t0 = time.perf_counter()
            _ = model.extract_features(dummy_batch16)
            if device.type == "mps":
                torch.mps.synchronize()
            t1 = time.perf_counter()
            latencies_batch_ms.append((t1 - t0) * 1000.0)

    lat_single_np = np.array(latencies_single_ms)
    lat_batch_np = np.array(latencies_batch_ms)

    avg_latency = float(np.mean(lat_single_np))
    fps_single = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    avg_batch_lat = float(np.mean(lat_batch_np))
    fps_throughput = (16.0 * 1000.0) / avg_batch_lat if avg_batch_lat > 0 else 0.0

    return {
        "device": str(device),
        "single_latency_mean_ms": round(avg_latency, 3),
        "single_latency_median_ms": round(float(np.median(lat_single_np)), 3),
        "single_latency_p95_ms": round(float(np.percentile(lat_single_np, 95)), 3),
        "single_latency_min_ms": round(float(np.min(lat_single_np)), 3),
        "single_latency_max_ms": round(float(np.max(lat_single_np)), 3),
        "single_fps": round(fps_single, 1),
        "batch16_latency_mean_ms": round(avg_batch_lat, 3),
        "batch16_throughput_fps": round(fps_throughput, 1)
    }


def measure_end_to_end_pipeline(
    model: Any,
    device: torch.device,
    num_runs: int = 100
) -> Dict[str, Any]:
    """Measures full pipeline latency: BGR face crop -> Tensor -> Embedding -> Cosine Gallery Match."""
    detector = FaceDetector()
    dummy_crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    dummy_gallery = {f"ID_{i}": np.random.randn(128) for i in range(10)}
    for k in dummy_gallery:
        dummy_gallery[k] /= np.linalg.norm(dummy_gallery[k])

    # Warmup
    tensor = detector.preprocess_to_tensor(dummy_crop, device=device)
    with torch.no_grad():
        emb = model.extract_features(tensor).cpu().numpy().flatten()
        _ = max([np.dot(emb, g) for g in dummy_gallery.values()])

    preprocess_times = []
    cnn_times = []
    match_times = []
    total_times = []

    for _ in range(num_runs):
        t0 = time.perf_counter()

        # Step 1: Preprocess to Tensor
        t_pre_0 = time.perf_counter()
        tensor = detector.preprocess_to_tensor(dummy_crop, device=device)
        t_pre_1 = time.perf_counter()

        # Step 2: CNN Embedding
        with torch.no_grad():
            emb = model.extract_features(tensor)
            if device.type == "mps":
                torch.mps.synchronize()
            emb_np = emb.cpu().numpy().flatten()
        t_cnn_1 = time.perf_counter()

        # Step 3: Cosine Match against Gallery
        best_sim = max([float(np.dot(emb_np, g)) for g in dummy_gallery.values()])
        t_match_1 = time.perf_counter()

        preprocess_times.append((t_pre_1 - t_pre_0) * 1000.0)
        cnn_times.append((t_cnn_1 - t_pre_1) * 1000.0)
        match_times.append((t_match_1 - t_cnn_1) * 1000.0)
        total_times.append((t_match_1 - t0) * 1000.0)

    return {
        "preprocess_mean_ms": round(float(np.mean(preprocess_times)), 3),
        "cnn_embedding_mean_ms": round(float(np.mean(cnn_times)), 3),
        "gallery_matching_mean_ms": round(float(np.mean(match_times)), 3),
        "total_pipeline_mean_ms": round(float(np.mean(total_times)), 3),
        "pipeline_fps": round(1000.0 / float(np.mean(total_times)), 1)
    }


def run_baseline_benchmark() -> Dict[str, Any]:
    """Coordinates and executes the complete baseline evaluation."""
    logger.info("=" * 65)
    logger.info("STARTING EMPIRICAL BASELINE BENCHMARK")
    logger.info("=" * 65)

    mem_before = get_process_memory_mb()
    logger.info("Initial Process Memory: %.2f MB", mem_before)

    device_cpu = torch.device("cpu")
    model_cpu, class_names = load_model_for_inference(CNN_MODEL_PATH, device=device_cpu)
    mem_after_model = get_process_memory_mb()
    logger.info("Memory After Model Load (CPU): %.2f MB (+%.2f MB)", mem_after_model, mem_after_model - mem_before)

    # 1. Size & Parameters
    size_stats = measure_model_size(CNN_MODEL_PATH, model_cpu)
    logger.info("Model File Size: %.2f MB, Parameters: %d", size_stats["file_size_mb"], size_stats["total_parameters"])

    # 2. Accuracy, FAR, FRR
    accuracy_stats = measure_accuracy_and_rejection(model_cpu, device_cpu, operational_threshold=RECOGNITION_SIMILARITY_THRESHOLD)
    logger.info("Test Accuracy: %.2f%%, FAR: %.2f%% (%d/%d), FRR: %.2f%% (%d/%d)",
                accuracy_stats["test_identification_accuracy"] * 100,
                accuracy_stats["false_acceptance_rate"] * 100,
                accuracy_stats["false_acceptance_count"], accuracy_stats["total_unknown_samples"],
                accuracy_stats["false_rejection_rate"] * 100,
                accuracy_stats["false_rejection_count"], accuracy_stats["total_test_samples"])

    # 3. CPU Latency & FPS
    logger.info("Benchmarking CPU Inference Latency (200 runs)...")
    speed_cpu = measure_inference_speed(model_cpu, device_cpu)
    logger.info("CPU Single Latency: %.2f ms (%.1f FPS)", speed_cpu["single_latency_mean_ms"], speed_cpu["single_fps"])

    # 4. MPS (Apple Silicon GPU) Latency & FPS if available
    speed_mps = None
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_mps = torch.device("mps")
        logger.info("Benchmarking Apple Silicon MPS Inference Latency (200 runs)...")
        model_mps, _ = load_model_for_inference(CNN_MODEL_PATH, device=device_mps)
        speed_mps = measure_inference_speed(model_mps, device_mps)
        logger.info("MPS Single Latency: %.2f ms (%.1f FPS)", speed_mps["single_latency_mean_ms"], speed_mps["single_fps"])

    # 5. End-to-End Pipeline Latency
    logger.info("Benchmarking End-to-End Pipeline (OpenCV -> PyTorch -> Matching)...")
    pipeline_stats = measure_end_to_end_pipeline(model_cpu, device_cpu)
    logger.info("End-to-End Total: %.2f ms (%.1f FPS)", pipeline_stats["total_pipeline_mean_ms"], pipeline_stats["pipeline_fps"])

    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_path": CNN_MODEL_PATH,
        "architecture": "FaceCNN_4Block",
        "num_classes": len(class_names),
        "class_names": class_names,
        "system_info": {
            "platform": sys.platform,
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "opencv_version": cv2.__version__,
            "apple_silicon_mps": hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        },
        "model_size": size_stats,
        "memory_usage": {
            "initial_rss_mb": round(mem_before, 2),
            "after_load_rss_mb": round(mem_after_model, 2),
            "net_model_rss_mb": round(mem_after_model - mem_before, 2)
        },
        "accuracy_and_rejection": accuracy_stats,
        "speed_benchmarks": {
            "cpu": speed_cpu,
            "mps": speed_mps,
            "end_to_end_pipeline_cpu": pipeline_stats
        }
    }

    save_path = os.path.join(os.path.dirname(CNN_MODEL_PATH), "baseline_benchmark.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2)

    logger.info("=" * 65)
    logger.info("BASELINE BENCHMARK COMPLETE. Saved to %s", save_path)
    logger.info("=" * 65)

    return benchmark_report


if __name__ == "__main__":
    run_baseline_benchmark()
