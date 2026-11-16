"""
config.py
---------
Centralized configuration parameters for the Multi-Class CNN Face Recognition
and RTSP CCTV Surveillance System.
"""

import os
from typing import Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# Paths & Directories
# ----------------------------------------------------------------------
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
CAPTURED_IMAGES_DIR = os.path.join(BASE_DIR, "captured_images")
UNKNOWN_TEST_DIR = os.path.join(BASE_DIR, "unknown_test")

CNN_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
METADATA_PATH = os.path.join(BASE_DIR, "person_metadata.json")
RECOGNITION_CONFIG_PATH = os.path.join(MODELS_DIR, "recognition_config.json")
HAAR_CASCADE_PATH = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")

METRICS_PLOT_PATH = os.path.join(BASE_DIR, "static", "images", "training_metrics.png")
CONFUSION_MATRIX_PATH = os.path.join(BASE_DIR, "static", "images", "confusion_matrix.png")

# Ensure critical directories exist
for path in [MODELS_DIR, DATASET_DIR, KNOWN_FACES_DIR, CAPTURED_IMAGES_DIR, UNKNOWN_TEST_DIR]:
    os.makedirs(path, exist_ok=True)

# ----------------------------------------------------------------------
# Face & CNN Architecture Settings
# ----------------------------------------------------------------------
IMAGE_SIZE = (112, 112)  # Standard input resolution for face recognition
CHANNELS = 3
BATCH_SIZE = 16
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 1e-4
DEFAULT_EPOCHS = 35
EARLY_STOPPING_PATIENCE = 8

# Image normalization matching standard ImageNet / Pretrained distributions
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# ----------------------------------------------------------------------
# Face Quality & Detection
# ----------------------------------------------------------------------
MIN_FACE_SIZE = (40, 40)
FACE_PADDING_RATIO = 0.15  # Expand face crop bounding box by 15% to include contour
MIN_LAPLACIAN_BLUR_VAR = 20.0  # Discard blurry faces below this variance threshold

# ----------------------------------------------------------------------
# Calibrated Decision & Metric Thresholds (Separated Architecture)
# ----------------------------------------------------------------------
# 1. YOLO Object Detection Confidence
YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE", "0.40"))

# 2. Face Detection Quality & Confidence
FACE_DETECTION_CONFIDENCE = float(os.environ.get("FACE_CONFIDENCE", "0.55"))

# 3. Face Recognition Similarity (Cosine Similarity with enrolled embeddings)
RECOGNITION_SIMILARITY_THRESHOLD = float(os.environ.get("RECOGNITION_THRESHOLD", "0.84"))

# 4. Identity Decision Threshold (Rejection boundary: < 0.84 -> UNKNOWN)
IDENTITY_DECISION_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.84"))
DEFAULT_CONFIDENCE_THRESHOLD = IDENTITY_DECISION_THRESHOLD
MAX_ALLOWED_FAR = 0.05  # 5% target False Acceptance Rate

# 5. Alert & Deduplication Policies
ALERT_ON_UNKNOWN_PERSON = os.environ.get("ALERT_ON_UNKNOWN", "true").lower() in ("true", "1", "yes")
ALERT_ON_ELECTRONIC = os.environ.get("ALERT_ON_ELECTRONIC", "true").lower() in ("true", "1", "yes")
ALERT_ON_VEHICLE = os.environ.get("ALERT_ON_VEHICLE", "false").lower() in ("true", "1", "yes")
SPATIAL_DEDUPLICATION_IOU = float(os.environ.get("SPATIAL_DEDUPLICATION_IOU", "0.35"))

# ----------------------------------------------------------------------
# Temporal Verification & Throttling
# ----------------------------------------------------------------------
# Process AI inference every N frames to conserve computational load
PROCESS_EVERY_N_FRAMES = int(os.environ.get("PROCESS_EVERY_N_FRAMES", "2"))
RECOGNITION_INTERVAL = int(os.environ.get("RECOGNITION_INTERVAL", "2"))

# Temporal window: require M consistent predictions out of last K frames
TEMPORAL_WINDOW_SIZE = int(os.environ.get("TEMPORAL_WINDOW_SIZE", "5"))
REQUIRED_CONSISTENT_FRAMES = int(os.environ.get("REQUIRED_CONSISTENT_FRAMES", "3"))

# Real-time visual debug HUD overlay
DEBUG_MODE = os.environ.get("DEBUG_RECOGNITION", "true").lower() in ("true", "1", "yes")

# Cooldown and tracking
DETECTION_COOLDOWN_SECONDS = float(os.environ.get("DETECTION_COOLDOWN_SECONDS", "60.0"))
MAX_MISSING_FRAMES = int(os.environ.get("MAX_MISSING_FRAMES", "30"))

