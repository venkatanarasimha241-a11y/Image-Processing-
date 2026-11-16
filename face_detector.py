"""
face_detector.py
----------------
Unified Face Detection, Quality Assessment, and Preprocessing Module.

Features:
1. Primary Detector: OpenCV YuNet Deep Learning Face Detector (FaceDetectorYN).
2. Secondary Fallback: Haar Cascade Classifier (if available in OpenCV build).
3. Tertiary Fallback: Anatomical upper head-region heuristic inside detected person bounding box.
4. Pre-inference Face Quality Gating: Evaluates resolution, sharpness (blur), lighting, and aspect ratio.
5. Standardized Tensor Preprocessing: 112x112 RGB normalized tensor for FaceCNN feature extractor.
"""

import os
import cv2
import logging
import threading
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any, Sequence, Union

from config import (
    IMAGE_SIZE, MIN_FACE_SIZE, FACE_PADDING_RATIO,
    MIN_LAPLACIAN_BLUR_VAR, NORM_MEAN, NORM_STD, HAAR_CASCADE_PATH,
    MODELS_DIR
)
from face_quality import evaluate_face_quality, FaceQualityAssessor

logger = logging.getLogger("FaceDetector")

YUNET_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")


def align_face_crop(
    image: np.ndarray,
    right_eye: Tuple[float, float],
    left_eye: Tuple[float, float]
) -> np.ndarray:
    """
    Rotates face image around the center between eyes so that the eye axis is horizontal.
    Only aligns if tilt angle is between -45 and 45 degrees.
    """
    if image is None or image.size == 0:
        return image
    dx = left_eye[0] - right_eye[0]
    dy = left_eye[1] - right_eye[1]
    dist = np.hypot(dx, dy)
    if dist < 6.0:
        return image
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if abs(angle) > 45.0 or abs(angle) < 0.8:
        return image
    eyes_center = ((right_eye[0] + left_eye[0]) / 2.0, (right_eye[1] + left_eye[1]) / 2.0)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
    aligned = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return aligned


class FaceDetector:
    def __init__(self, cascade_path: str = HAAR_CASCADE_PATH):
        self.cascade_path = cascade_path
        self.yunet_detector = None
        self.face_cascade = None
        self._detector_lock = threading.Lock()
        self.quality_assessor = FaceQualityAssessor(
            min_size=MIN_FACE_SIZE,
            min_blur_var=MIN_LAPLACIAN_BLUR_VAR
        )

        # Precompute normalization tensors
        self.mean_tensor = torch.tensor(NORM_MEAN, dtype=torch.float32).view(3, 1, 1)
        self.std_tensor = torch.tensor(NORM_STD, dtype=torch.float32).view(3, 1, 1)

        self._init_detectors()

    def _init_detectors(self):
        """Initializes YuNet first, then Haar cascade as fallback."""
        # 1. Try YuNet (Deep Learning detector in OpenCV)
        if os.path.exists(YUNET_MODEL_PATH) and hasattr(cv2, "FaceDetectorYN_create"):
            try:
                create_fn = getattr(cv2, "FaceDetectorYN_create")
                self.yunet_detector = create_fn(
                    YUNET_MODEL_PATH,
                    "",
                    (320, 320),
                    score_threshold=0.35,
                    nms_threshold=0.3
                )
                logger.info("FaceDetector: Loaded YuNet deep learning detector from %s", YUNET_MODEL_PATH)
            except Exception as e:
                logger.warning("Could not initialize YuNet: %s", e)
                self.yunet_detector = None

        # 2. Try Haar Cascade if available
        cascade_fn = getattr(cv2, "CascadeClassifier", None)
        if cascade_fn is not None:
            candidates = [
                self.cascade_path,
                os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml"),
                getattr(getattr(cv2, "data", None), "haarcascades", "") + "haarcascade_frontalface_default.xml"
                if hasattr(cv2, "data") else "",
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    try:
                        self.face_cascade = cascade_fn(p)
                        if not self.face_cascade.empty():
                            logger.info("FaceDetector: Loaded Haar Cascade from %s", p)
                            break
                    except Exception:
                        pass

    def detect_face_in_person(
        self,
        frame: np.ndarray,
        person_bbox: Sequence[Union[int, float]]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], float, bool]:
        """
        Extracts face from a person bounding box [x1, y1, x2, y2].
        Returns:
            face_crop (BGR np.ndarray or None)
            face_box ([fx1, fy1, fx2, fy2] in global frame coordinates or None)
            face_confidence (float [0.0, 1.0])
            is_valid_face (bool)
        """
        if frame is None or frame.size == 0 or not person_bbox:
            return None, None, 0.0, False

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in person_bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cropped_person = frame[y1:y2, x1:x2]
        if cropped_person.size == 0:
            return None, None, 0.0, False

        ph, pw = cropped_person.shape[:2]

        # 0. Fast-path: If the crop is already a normalized face avatar / headshot (<= 160px with roughly square ratio),
        # return directly without double-cropping chin/hair.
        if ph <= 160 and pw <= 160 and 0.78 <= (ph / float(max(1, pw))) <= 1.28:
            if float(np.std(cropped_person)) >= 14.0:
                global_box = [x1, y1, x2, y2]
                return cropped_person, global_box, 0.90, True

        # 1. Try YuNet deep detector inside person crop
        if self.yunet_detector is not None:
            try:
                # Robust scaling for YuNet: upscale small crops to >= 320, downscale large to <= 480
                scale = 1.0
                if max(ph, pw) > 480:
                    scale = 480.0 / float(max(ph, pw))
                elif max(ph, pw) < 240 and min(ph, pw) > 20:
                    scale = min(3.0, 320.0 / float(max(ph, pw)))

                if abs(scale - 1.0) > 0.01:
                    inp = cv2.resize(cropped_person, (int(pw * scale), int(ph * scale)))
                else:
                    inp = cropped_person

                ih, iw = inp.shape[:2]

                with self._detector_lock:
                    self.yunet_detector.setInputSize((iw, ih))
                    _, faces = self.yunet_detector.detect(inp)

                if faces is not None and len(faces) > 0:
                    # Pick face with highest confidence
                    best_face = max(faces, key=lambda f: float(f[-1]))
                    conf = float(best_face[-1])
                    if conf >= 0.35:
                        raw_bbox = np.array(best_face[0:4], dtype=np.float32) / scale
                        fx, fy, fw, fh = [int(v) for v in raw_bbox]

                        # Apply padding
                        pad_w = int(fw * FACE_PADDING_RATIO)
                        pad_h = int(fh * FACE_PADDING_RATIO)
                        p_fx1 = max(0, fx - pad_w)
                        p_fy1 = max(0, fy - pad_h)
                        p_fx2 = min(pw, fx + fw + pad_w)
                        p_fy2 = min(ph, fy + fh + pad_h)

                        face_crop = cropped_person[p_fy1:p_fy2, p_fx1:p_fx2]
                        if face_crop.size > 0:
                            # Align face using eye landmarks if available
                            if len(best_face) >= 14:
                                re_x = float(best_face[4] / scale) - p_fx1
                                re_y = float(best_face[5] / scale) - p_fy1
                                le_x = float(best_face[6] / scale) - p_fx1
                                le_y = float(best_face[7] / scale) - p_fy1
                                face_crop = align_face_crop(face_crop, (re_x, re_y), (le_x, le_y))

                            global_box = [x1 + p_fx1, y1 + p_fy1, x1 + p_fx2, y1 + p_fy2]
                            return face_crop, global_box, conf, True
            except Exception as e:
                logger.debug("YuNet detection error: %s", e)

        # 2. Try Haar cascade inside person crop
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(cropped_person, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=MIN_FACE_SIZE
                )
                if len(faces) > 0:
                    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                    fx, fy, fw, fh = faces[0]
                    pad_w = int(fw * FACE_PADDING_RATIO)
                    pad_h = int(fh * FACE_PADDING_RATIO)
                    p_fx1 = max(0, fx - pad_w)
                    p_fy1 = max(0, fy - pad_h)
                    p_fx2 = min(pw, fx + fw + pad_w)
                    p_fy2 = min(ph, fy + fh + pad_h)

                    face_crop = cropped_person[p_fy1:p_fy2, p_fx1:p_fx2]
                    global_box = [x1 + p_fx1, y1 + p_fy1, x1 + p_fx2, y1 + p_fy2]
                    return face_crop, global_box, 0.85, True
            except Exception:
                pass

        # 3. Fallback: Only if the cropped region is ALREADY a headshot/portrait photo
        # (aspect ratio roughly square, e.g. uploaded photo or pre-cropped face avatar)
        if ph >= MIN_FACE_SIZE[1] and pw >= MIN_FACE_SIZE[0]:
            aspect_ratio = ph / float(max(1, pw))
            if 0.75 <= aspect_ratio <= 1.35:
                # Validate texture so it's not a uniform color wall or blank rectangle
                if float(np.std(cropped_person)) >= 15.0:
                    global_box = [x1, y1, x2, y2]
                    return cropped_person, global_box, 0.70, True

        # If person is seen in CCTV but no face detected (e.g. facing away, scalp only, occlusion):
        # Decisively return NO FACE rather than returning scalp hair
        return None, None, 0.0, False

    def detect_all_faces(self, image_bgr: np.ndarray, min_conf: float = 0.40) -> List[Dict[str, Any]]:
        """
        Detects all faces in an image using YuNet (or Haar cascade fallback).
        Returns a list of dicts: [{"crop": np.ndarray, "bbox": [x1, y1, x2, y2], "confidence": float}].
        """
        if image_bgr is None or image_bgr.size == 0:
            return []

        h, w = image_bgr.shape[:2]
        detected: List[Dict[str, Any]] = []

        # 1. Try YuNet deep learning detector
        if self.yunet_detector is not None:
            try:
                scale = min(1.0, 640.0 / max(h, w)) if max(h, w) > 640 else 1.0
                inp = cv2.resize(image_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else image_bgr
                ih, iw = inp.shape[:2]
                with self._detector_lock:
                    self.yunet_detector.setInputSize((iw, ih))
                    _, faces = self.yunet_detector.detect(inp)
                if faces is not None and len(faces) > 0:
                    for f in faces:
                        conf = float(f[-1])
                        if conf >= min_conf:
                            raw_bbox = np.array(f[0:4], dtype=np.float32) / scale
                            fx, fy, fw, fh = [int(v) for v in raw_bbox]
                            pad_w = int(fw * FACE_PADDING_RATIO)
                            pad_h = int(fh * FACE_PADDING_RATIO)
                            x1 = max(0, fx - pad_w)
                            y1 = max(0, fy - pad_h)
                            x2 = min(w, fx + fw + pad_w)
                            y2 = min(h, fy + fh + pad_h)
                            crop = image_bgr[y1:y2, x1:x2]
                            if crop.size > 0:
                                if len(f) >= 14:
                                    re_x = float(f[4] / scale) - x1
                                    re_y = float(f[5] / scale) - y1
                                    le_x = float(f[6] / scale) - x1
                                    le_y = float(f[7] / scale) - y1
                                    crop = align_face_crop(crop, (re_x, re_y), (le_x, le_y))
                                detected.append({
                                    "crop": crop,
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": conf
                                })
                    if detected:
                        return detected
            except Exception as e:
                logger.debug("YuNet detect_all_faces error: %s", e)

        # 2. Try Haar cascade
        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=MIN_FACE_SIZE
                )
                if len(faces) > 0:
                    for (fx, fy, fw, fh) in faces:
                        pad_w = int(fw * FACE_PADDING_RATIO)
                        pad_h = int(fh * FACE_PADDING_RATIO)
                        x1 = max(0, fx - pad_w)
                        y1 = max(0, fy - pad_h)
                        x2 = min(w, fx + fw + pad_w)
                        y2 = min(h, fy + fh + pad_h)
                        crop = image_bgr[y1:y2, x1:x2]
                        if crop.size > 0:
                            detected.append({
                                "crop": crop,
                                "bbox": [x1, y1, x2, y2],
                                "confidence": 0.85
                            })
                    return detected
            except Exception as e:
                logger.debug("Haar detect_all_faces error: %s", e)

        # 3. Fallback: If image itself is nearly square portrait and passes texture check
        if h >= MIN_FACE_SIZE[1] and w >= MIN_FACE_SIZE[0]:
            aspect_ratio = h / float(max(1, w))
            if 0.7 <= aspect_ratio <= 1.4:
                if float(np.std(image_bgr)) >= 15.0:
                    detected.append({
                        "crop": image_bgr,
                        "bbox": [0, 0, w, h],
                        "confidence": 0.65
                    })

        return detected

    def is_face_acceptable(self, face_bgr: np.ndarray) -> Tuple[bool, str, float]:
        """Validates face crop using dedicated face quality assessor."""
        result = self.quality_assessor.evaluate(face_bgr)
        reason = ", ".join(result["issues"]) if result["issues"] else "Acceptable"
        return result["is_acceptable"], reason, result["blur_score"]

    def calculate_blur_variance(self, face_bgr: np.ndarray) -> float:
        """Calculates Laplacian variance of a face image (delegated to quality_assessor)."""
        return self.quality_assessor.calculate_blur_variance(face_bgr)

    def preprocess_face(self, face_bgr: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
        """Resizes cleanly and converts BGR to RGB."""
        if face_bgr is None or face_bgr.size == 0:
            raise ValueError("Empty face image provided")
        resized = cv2.resize(face_bgr, target_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    def preprocess_to_tensor(self, face_bgr: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Converts BGR face crop into standardized (1, 3, 112, 112) normalized PyTorch tensor."""
        rgb = self.preprocess_face(face_bgr, target_size=IMAGE_SIZE)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        tensor = (tensor - self.mean_tensor) / self.std_tensor
        return tensor.unsqueeze(0).to(device)
