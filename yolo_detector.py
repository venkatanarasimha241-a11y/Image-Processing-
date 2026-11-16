"""
yolo_detector.py
----------------
Ultralytics YOLO Object Detector with Multi-Class Detection, Class-Specific Thresholds,
Synonym Aliasing, Non-Maximum Suppression, and Bounding-Box Validation.

Supported Categories include:
- Person
- Mobile phone / Cell phone
- Laptop
- Backpack
- Bottle
- Chair
- Car
- Motorcycle
- Dog
- Other supported COCO objects (80 classes)
"""

import os
import cv2
import logging
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

logger = logging.getLogger("YOLODetector")

# Default settings from environment
DEFAULT_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
DEFAULT_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.40"))
DEFAULT_IOU_THRESHOLD = float(os.environ.get("YOLO_IOU", "0.45"))

# Synonym mappings: maps user-friendly names to official COCO class names
CLASS_ALIASES = {
    "mobile phone": "cell phone",
    "mobile": "cell phone",
    "phone": "cell phone",
    "smartphone": "cell phone",
    "cellphone": "cell phone",
    "computer": "laptop",
    "notebook": "laptop",
    "bag": "backpack",
    "schoolbag": "backpack",
    "motorbike": "motorcycle",
    "bike": "bicycle",
    "automobile": "car",
    "canine": "dog"
}

# Standard required surveillance categories
REQUIRED_CATEGORIES = {
    "person", "cell phone", "laptop", "backpack", "bottle",
    "chair", "car", "motorcycle", "dog", "bus", "truck", "bicycle"
}

# Class-specific confidence thresholds to reduce false positives while detecting small items
CLASS_THRESHOLDS = {
    "cell phone": 0.20,   # Lower threshold for small/occluded CCTV phone detections
    "bottle": 0.30,
    "backpack": 0.32,
    "laptop": 0.35,
    "chair": 0.35,
    "person": 0.40,
    "dog": 0.40,
    "car": 0.42,
    "motorcycle": 0.40,
    "bus": 0.42,
    "truck": 0.42,
    "bicycle": 0.38
}


def normalize_class_name(name: str) -> str:
    """Normalizes class name using alias dictionary."""
    clean = name.strip().lower()
    return CLASS_ALIASES.get(clean, clean)


class YOLODetector:
    """
    Robust Object Detection Pipeline using Ultralytics YOLOv8.
    Provides NMS, bounding box validation, class filtering, and telemetry.
    """
    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD
    ):
        self.requested_model_path = model_path
        self.active_model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        self.model = None
        self.is_loaded = False
        self.status_message = "Initializing..."
        self.class_names: Dict[int, str] = {}
        self.allowed_classes: Optional[Set[str]] = None
        self.custom_model_warning: Optional[str] = None

        # Resolve configured allowed classes from environment (or default to required set)
        raw_allowed = os.environ.get("YOLO_ALLOWED_CLASSES", "")
        if raw_allowed.strip():
            self.allowed_classes = {normalize_class_name(c) for c in raw_allowed.split(",") if c.strip()}
        else:
            # By default allow the required categories plus common COCO objects
            self.allowed_classes = set(REQUIRED_CATEGORIES)

        self._load_model()

    def _load_model(self):
        """
        Loads the configured YOLO model.
        If a custom path (e.g., models/best.pt) is missing or doesn't support required classes,
        falls back to yolov8n.pt with detailed diagnostics.
        """
        try:
            from ultralytics import YOLO

            model_to_try = self.requested_model_path

            # Check if custom model file exists
            if os.path.exists(model_to_try) and os.path.getsize(model_to_try) > 1000:
                logger.info("Loading specified YOLO weights from: %s", model_to_try)
                self.model = YOLO(model_to_try)
                self.active_model_path = model_to_try
                self.is_loaded = True

                # Inspect classes in custom model
                if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                    self.class_names = self.model.names
                    trained_classes = set(v.strip().lower() for v in self.model.names.values())
                    logger.info("Custom model loaded with %d classes: %s", len(trained_classes), trained_classes)

                    # Check for missing critical classes (e.g. mobile phone, laptop)
                    missing_reqs = [req for req in ["person", "cell phone"] if req not in trained_classes]
                    if missing_reqs:
                        self.custom_model_warning = (
                            f"Custom model '{os.path.basename(model_to_try)}' lacks trained classes: {missing_reqs}. "
                            f"Consider using standard 'yolov8n.pt' (which has 80 COCO classes including cell phone) "
                            f"or fine-tuning your custom model with annotated samples."
                        )
                        logger.warning(self.custom_model_warning)

                self.status_message = f"Loaded model: {os.path.basename(self.active_model_path)} ({len(self.class_names)} classes)"
                return

            # Fallback to standard yolov8n.pt which contains all 80 COCO classes
            fallback = "yolov8n.pt"
            logger.info("Custom weights not found at '%s'. Loading pre-trained '%s'...", model_to_try, fallback)
            self.model = YOLO(fallback)
            self.active_model_path = fallback
            self.is_loaded = True
            if hasattr(self.model, "names") and isinstance(self.model.names, dict):
                self.class_names = self.model.names

            self.status_message = f"Pretrained YOLOv8n Active (80 COCO classes including mobile phone, laptop, person)"
            logger.info("Pretrained YOLOv8n successfully loaded (%d classes)", len(self.class_names))

        except Exception as e:
            logger.error("Failed to load YOLO model: %s", e)
            self.model = None
            self.is_loaded = False
            self.status_message = f"YOLO load error: {str(e)}"

    def set_confidence(self, new_conf: float):
        """Updates base detection confidence threshold."""
        self.confidence_threshold = max(0.1, min(0.95, float(new_conf)))
        logger.info("YOLO confidence threshold updated to: %.2f", self.confidence_threshold)

    def set_allowed_classes(self, classes_list: List[str]):
        """Updates the list of filtered detection classes."""
        self.allowed_classes = {normalize_class_name(c) for c in classes_list if c.strip()}
        logger.info("YOLO allowed classes updated to: %s", self.allowed_classes)

    def detect(self, frame: Optional[np.ndarray], is_single_image: bool = False) -> List[Dict[str, Any]]:
        """
        Executes YOLO inference on a single OpenCV BGR image frame.
        Performs NMS, bounding box validation, class mapping, and threshold checking.

        Returns list of detection dictionaries:
        [
            {
                "class_id": 0,
                "class_name": "mobile phone",
                "confidence": 0.912,
                "bbox": [x1, y1, x2, y2],
                "track_id": 1,
                "is_vehicle": False,
                "is_person": False,
                "is_electronic": True
            }
        ]
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return []

        if not self.is_loaded or self.model is None:
            return []

        h, w = frame.shape[:2]

        try:
            # For single uploaded image inference, use direct predict to avoid tracker latency/filtering
            if is_single_image:
                predict_kwargs = {
                    "verbose": False,
                    "conf": min(self.confidence_threshold, 0.22),
                    "iou": self.iou_threshold
                }
                results = self.model(frame, **predict_kwargs)
            else:
                track_kwargs = {
                    "persist": True,
                    "verbose": False,
                    "conf": min(self.confidence_threshold, 0.25),
                    "iou": self.iou_threshold,
                    "tracker": "bytetrack.yaml"
                }

                try:
                    results = self.model.track(frame, **track_kwargs)
                except Exception as track_err:
                    logger.debug("Tracker fallback to predict: %s", track_err)
                    predict_kwargs = {
                        "verbose": False,
                        "conf": min(self.confidence_threshold, 0.25),
                        "iou": self.iou_threshold
                    }
                    results = self.model(frame, **predict_kwargs)

            detections: List[Dict[str, Any]] = []

            for r in results:
                boxes = getattr(r, "boxes", None)
                if boxes is None:
                    continue

                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [int(v) for v in xyxy]

                    # Extract tracking ID if available
                    track_id = None
                    if hasattr(box, "id") and box.id is not None:
                        try:
                            track_id = int(box.id[0].item())
                        except Exception:
                            track_id = None

                    # Bounding Box Validation: Clamping & Boundary checks
                    x1 = max(0, min(w - 1, x1))
                    y1 = max(0, min(h - 1, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Resolve class name
                    raw_name = self.class_names.get(cls_id, f"class_{cls_id}")
                    raw_str = raw_name.strip().lower() if isinstance(raw_name, str) else str(raw_name).strip().lower()
                    canonical_name = normalize_class_name(raw_str)

                    # Phones can be tiny in wide-angle CCTV frames. Use a smaller minimum
                    # for phone boxes; keep stricter validation for other object classes.
                    if canonical_name == "cell phone":
                        if box_w < 6 or box_h < 6 or (box_w * box_h) < 36:
                            continue
                    elif box_w < 12 or box_h < 12 or (box_w * box_h) < 150:
                        continue

                    # Class filtering
                    if self.allowed_classes is not None and canonical_name not in self.allowed_classes and raw_str not in self.allowed_classes:
                        continue

                    # Class-specific confidence gating
                    required_conf = CLASS_THRESHOLDS.get(canonical_name, self.confidence_threshold)
                    if conf < required_conf:
                        continue

                    # Reject only extremely implausible phone boxes. In close-up or
                    # cropped images a real phone can occupy a large part of the frame.
                    if canonical_name == "cell phone":
                        if (box_h / float(h) > 0.70) or ((box_w * box_h) / float(w * h) > 0.40):
                            logger.debug("Rejected implausibly oversized cell phone detection: %dx%d in %dx%d frame", box_w, box_h, w, h)
                            continue

                    # Determine semantic categories
                    is_vehicle = canonical_name in {"car", "motorcycle", "bus", "truck", "bicycle"}
                    is_person = canonical_name == "person"
                    is_electronic = canonical_name in {"cell phone", "laptop", "tv", "mouse", "keyboard"}

                    # Display name: display friendly name for cell phone -> "mobile phone"
                    display_name = "mobile phone" if canonical_name == "cell phone" else canonical_name

                    detections.append({
                        "class_id": cls_id,
                        "class_name": display_name,
                        "raw_class_name": raw_str,
                        "canonical_class": canonical_name,
                        "confidence": round(conf, 3),
                        "bbox": [x1, y1, x2, y2],
                        "track_id": track_id,
                        "is_vehicle": is_vehicle,
                        "is_person": is_person,
                        "is_electronic": is_electronic
                    })

            # Post-process: Resolve conflicts where a person was mistakenly dual-detected as a whole-body phone
            person_boxes = [d["bbox"] for d in detections if d["is_person"]]
            clean_detections = []
            for d in detections:
                if d["canonical_class"] == "cell phone" and person_boxes:
                    px1, py1, px2, py2 = d["bbox"]
                    phone_area = (px2 - px1) * (py2 - py1)
                    is_spurious_phone = False
                    for bx1, by1, bx2, by2 in person_boxes:
                        # Compute intersection
                        ix1, iy1 = max(px1, bx1), max(py1, by1)
                        ix2, iy2 = min(px2, bx2), min(py2, by2)
                        if ix2 > ix1 and iy2 > iy1:
                            inter_area = (ix2 - ix1) * (iy2 - iy1)
                            # A legitimate handheld phone normally overlaps its holder's
                            # person box. Suppress only near-body-sized phone boxes, not normal
                            # small phone boxes contained inside a person detection.
                            person_area = (bx2 - bx1) * (by2 - by1)
                            if person_area > 0 and phone_area > 0:
                                phone_to_person = phone_area / float(person_area)
                                phone_overlap = inter_area / float(phone_area)
                                if phone_to_person > 0.65 and phone_overlap > 0.90:
                                    is_spurious_phone = True
                                    break
                    if is_spurious_phone:
                        logger.debug("Suppressed spurious phone detection overlapping body")
                        continue
                clean_detections.append(d)

            return clean_detections

        except Exception as e:
            logger.error("YOLO detection error: %s", e)
            return []

    def get_status(self) -> Dict[str, Any]:
        """Returns diagnostic status of the detector."""
        return {
            "is_loaded": self.is_loaded,
            "active_model_path": self.active_model_path,
            "requested_model_path": self.requested_model_path,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold,
            "classes_count": len(self.class_names),
            "allowed_classes": sorted(list(self.allowed_classes)) if self.allowed_classes else "all",
            "status_message": self.status_message,
            "custom_model_warning": self.custom_model_warning
        }
