"""
face_quality.py
---------------
Dedicated Face Quality Assessment and Validation Module.
Performs pre-inference gating to discard degraded, blurry, or low-resolution face crops:
1. Minimum face resolution (width & height)
2. Sharpness / Blur detection using Laplacian variance
3. Illumination / Brightness & Contrast analysis
4. Facial aspect ratio validation
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger("FaceQuality")

# Quality defaults
DEFAULT_MIN_SIZE = (36, 36)
DEFAULT_MIN_LAPLACIAN_VAR = 18.0
DEFAULT_MIN_BRIGHTNESS = 30.0
DEFAULT_MAX_BRIGHTNESS = 230.0
DEFAULT_MIN_CONTRAST = 15.0
DEFAULT_MIN_ASPECT_RATIO = 0.55
DEFAULT_MAX_ASPECT_RATIO = 1.75


class FaceQualityAssessor:
    """
    Evaluates face crops before passing them to the deep recognition model.
    Helps eliminate false recognitions caused by motion blur, extreme darkness,
    or partial crops.
    """
    def __init__(
        self,
        min_size: Tuple[int, int] = DEFAULT_MIN_SIZE,
        min_blur_var: float = DEFAULT_MIN_LAPLACIAN_VAR,
        min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
        max_brightness: float = DEFAULT_MAX_BRIGHTNESS,
        min_contrast: float = DEFAULT_MIN_CONTRAST
    ):
        self.min_size = min_size
        self.min_blur_var = min_blur_var
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.min_contrast = min_contrast

    def calculate_blur_variance(self, face_bgr: np.ndarray) -> float:
        """Calculates Laplacian variance. Higher value = sharper image."""
        if face_bgr is None or face_bgr.size == 0:
            return 0.0
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception as e:
            logger.debug("Error computing blur variance: %s", e)
            return 0.0

    def calculate_illumination(self, face_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Calculates average brightness and contrast (standard deviation of grayscale intensities).
        Returns: (mean_brightness, contrast)
        """
        if face_bgr is None or face_bgr.size == 0:
            return 0.0, 0.0
        try:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY) if len(face_bgr.shape) == 3 else face_bgr
            mean_val = float(np.mean(gray))
            contrast_val = float(np.std(gray))
            return mean_val, contrast_val
        except Exception:
            return 0.0, 0.0

    def check_aspect_ratio(self, face_bgr: np.ndarray) -> Tuple[bool, float]:
        """Validates that face crop is roughly facial proportions (width / height)."""
        if face_bgr is None or face_bgr.size == 0:
            return False, 0.0
        h, w = face_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return False, 0.0
        ratio = float(w) / float(h)
        is_valid = DEFAULT_MIN_ASPECT_RATIO <= ratio <= DEFAULT_MAX_ASPECT_RATIO
        return is_valid, ratio

    def evaluate(self, face_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive quality assessment of a face crop.
        Returns:
            {
                "is_acceptable": bool,
                "overall_score": float [0.0, 1.0],
                "blur_score": float,
                "brightness": float,
                "contrast": float,
                "width": int,
                "height": int,
                "aspect_ratio": float,
                "issues": List[str]
            }
        """
        if face_bgr is None or not isinstance(face_bgr, np.ndarray) or face_bgr.size == 0:
            return {
                "is_acceptable": False,
                "overall_score": 0.0,
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "width": 0,
                "height": 0,
                "aspect_ratio": 0.0,
                "issues": ["Empty face crop"]
            }

        h, w = face_bgr.shape[:2]
        issues: List[str] = []

        # 1. Size check
        if w < self.min_size[0] or h < self.min_size[1]:
            issues.append(f"Face too small ({w}x{h} < {self.min_size[0]}x{self.min_size[1]})")

        # 2. Blur / sharpness check
        blur_score = self.calculate_blur_variance(face_bgr)
        if blur_score < self.min_blur_var:
            issues.append(f"Face blurry (var {blur_score:.1f} < {self.min_blur_var:.1f})")

        # 3. Illumination check
        brightness, contrast = self.calculate_illumination(face_bgr)
        if brightness < self.min_brightness:
            issues.append(f"Underexposed / too dark ({brightness:.1f} < {self.min_brightness})")
        elif brightness > self.max_brightness:
            issues.append(f"Overexposed / washed out ({brightness:.1f} > {self.max_brightness})")

        if contrast < self.min_contrast:
            issues.append(f"Low contrast ({contrast:.1f} < {self.min_contrast})")

        # 4. Aspect ratio check
        aspect_valid, aspect_ratio = self.check_aspect_ratio(face_bgr)
        if not aspect_valid:
            issues.append(f"Distorted aspect ratio ({aspect_ratio:.2f})")

        # Compute normalized composite quality score in [0.0, 1.0]
        # Blur component (saturates at 100.0)
        blur_comp = min(1.0, blur_score / 80.0)
        # Size component (saturates at 150x150)
        size_comp = min(1.0, (w * h) / (120.0 * 120.0))
        # Illumination component (optimal around 110-150)
        illum_comp = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        # Contrast component
        contrast_comp = min(1.0, contrast / 50.0)

        overall_score = round(
            0.35 * blur_comp + 0.30 * size_comp + 0.20 * illum_comp + 0.15 * contrast_comp,
            3
        )

        is_acceptable = len(issues) == 0

        return {
            "is_acceptable": is_acceptable,
            "overall_score": overall_score,
            "blur_score": round(blur_score, 1),
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "width": w,
            "height": h,
            "aspect_ratio": round(aspect_ratio, 2),
            "issues": issues
        }


# Global default instance
_default_assessor = FaceQualityAssessor()

def evaluate_face_quality(face_bgr: np.ndarray) -> Dict[str, Any]:
    """Helper function to assess quality using global default assessor."""
    return _default_assessor.evaluate(face_bgr)
