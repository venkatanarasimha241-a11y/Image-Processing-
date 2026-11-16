"""
test_yolo_detector.py
---------------------
Unit tests for YOLO Object Detection pipeline (yolo_detector.py).
Covers all 12 specified scenarios with mocked YOLO models, avoiding
physical camera, GPU, or model weight downloads.
"""

import os
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch

from yolo_detector import YOLODetector, normalize_class_name, CLASS_ALIASES


from typing import Optional, cast

class MockBox:
    """Mock Ultralytics Box object."""
    def __init__(self, cls_id: int, conf: float, xyxy: list, track_id: Optional[int] = None):
        self.cls = torch.tensor([cls_id])
        self.conf = torch.tensor([conf])
        self.xyxy = torch.tensor([xyxy])
        self.id = torch.tensor([track_id]) if track_id is not None else None


class MockResult:
    """Mock Ultralytics Results object containing detection boxes."""
    def __init__(self, boxes: list):
        self.boxes = boxes


@pytest.fixture
def mock_yolo_backend():
    """Mocks ultralytics YOLO class constructor and model instance."""
    with patch("ultralytics.YOLO") as mock_cls:
        mock_model = MagicMock()
        mock_model.names = {0: "person", 67: "cell phone", 63: "laptop", 2: "car"}
        mock_cls.return_value = mock_model
        yield mock_model


class TestYOLODetector:

    def test_01_successful_model_initialization(self, mock_yolo_backend):
        """Scenario 1: Successful model initialization."""
        detector = YOLODetector(model_path="yolov8n.pt", confidence_threshold=0.40)
        assert detector.is_loaded is True
        assert detector.model is not None
        assert 0 in detector.class_names
        assert detector.class_names[0] == "person"

    def test_02_missing_model_weights_fallback(self):
        """Scenario 2: Missing custom model weights falls back gracefully to standard weights."""
        with patch("ultralytics.YOLO") as mock_cls, \
             patch("os.path.exists", return_value=False):
            mock_model = MagicMock()
            mock_model.names = {0: "person"}
            mock_cls.return_value = mock_model

            detector = YOLODetector(model_path="nonexistent_custom_model.pt")
            assert detector.is_loaded is True
            # Should fall back to yolov8n.pt
            assert detector.active_model_path == "yolov8n.pt"

    def test_03_invalid_image_input(self, mock_yolo_backend):
        """Scenario 3: Invalid image input (e.g. None or non-array)."""
        detector = YOLODetector()
        assert detector.detect(None) == []
        assert detector.detect(cast(np.ndarray, "not_an_image")) == []
        assert detector.detect(cast(np.ndarray, 12345)) == []

    def test_04_empty_image_input(self, mock_yolo_backend):
        """Scenario 4: Empty image input (e.g. array of size 0)."""
        detector = YOLODetector()
        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        assert detector.detect(empty_img) == []

    def test_05_successful_person_detection(self, mock_yolo_backend):
        """Scenario 5: Successful person detection."""
        detector = YOLODetector()
        mock_box = MockBox(cls_id=0, conf=0.85, xyxy=[50, 40, 200, 350], track_id=1)
        mock_yolo_backend.return_value = [MockResult(boxes=[mock_box])]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)

        assert len(dets) == 1
        d = dets[0]
        assert d["is_person"] is True
        assert d["class_name"] == "person"
        assert d["confidence"] == 0.85
        assert d["bbox"] == [50, 40, 200, 350]

    def test_06_no_detections_in_image(self, mock_yolo_backend):
        """Scenario 6: No detections returned by model."""
        detector = YOLODetector()
        mock_yolo_backend.return_value = [MockResult(boxes=[])]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)
        assert dets == []

    def test_07_filtering_detections_by_confidence_threshold(self, mock_yolo_backend):
        """Scenario 7: Filtering detections below configured confidence threshold."""
        detector = YOLODetector(confidence_threshold=0.50)
        # Person with confidence 0.35 (below 0.50 and class threshold 0.40)
        low_conf_box = MockBox(cls_id=0, conf=0.35, xyxy=[10, 10, 100, 200])
        # Person with confidence 0.85 (above threshold)
        high_conf_box = MockBox(cls_id=0, conf=0.85, xyxy=[120, 10, 220, 200])

        mock_yolo_backend.return_value = [MockResult(boxes=[low_conf_box, high_conf_box])]
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)

        assert len(dets) == 1
        assert dets[0]["confidence"] == 0.85

    def test_08_correct_bounding_box_clamping_and_boundaries(self, mock_yolo_backend):
        """Scenario 8: Correct bounding-box coordinate handling (clamped within frame bounds)."""
        detector = YOLODetector()
        # Box extending outside [0, 0, 640, 480]
        out_of_bounds_box = MockBox(cls_id=0, conf=0.90, xyxy=[-50, -20, 700, 550])
        mock_yolo_backend.return_value = [MockResult(boxes=[out_of_bounds_box])]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)

        assert len(dets) == 1
        x1, y1, x2, y2 = dets[0]["bbox"]
        assert x1 >= 0
        assert y1 >= 0
        assert x2 <= 640
        assert y2 <= 480

    def test_09_correct_class_label_mapping_and_synonym_aliasing(self, mock_yolo_backend):
        """Scenario 9: Correct class-label mapping and display name aliasing."""
        detector = YOLODetector()
        # cell phone class (id 67) -> canonical "cell phone" -> display "mobile phone"
        phone_box = MockBox(cls_id=67, conf=0.75, xyxy=[20, 20, 80, 120])
        mock_yolo_backend.return_value = [MockResult(boxes=[phone_box])]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)

        assert len(dets) == 1
        assert dets[0]["class_name"] == "mobile phone"
        assert dets[0]["canonical_class"] == "cell phone"
        assert dets[0]["is_electronic"] is True

        # Test synonym helper
        assert normalize_class_name("smartphone") == "cell phone"
        assert normalize_class_name("phone") == "cell phone"
        assert normalize_class_name("automobile") == "car"

    def test_10_handling_of_malformed_model_outputs(self, mock_yolo_backend):
        """Scenario 10: Handling of malformed model outputs without crashing."""
        detector = YOLODetector()
        # Malformed result with boxes=None or non-iterable
        bad_result = MagicMock()
        bad_result.boxes = None
        mock_yolo_backend.return_value = [bad_result]

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)
        assert dets == []

    def test_11_handling_of_model_inference_exceptions(self, mock_yolo_backend):
        """Scenario 11: Handling of exceptions during model inference."""
        detector = YOLODetector()
        mock_yolo_backend.side_effect = RuntimeError("CUDA/MPS internal execution error")

        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame, is_single_image=True)
        # Should gracefully catch error and return empty list rather than crashing server
        assert dets == []

    def test_12_behavior_when_model_is_unavailable(self):
        """Scenario 12: Correct behavior when YOLO model is unavailable."""
        with patch("ultralytics.YOLO", side_effect=Exception("Weights not found or corrupted")):
            detector = YOLODetector()
            assert detector.is_loaded is False
            assert detector.model is None
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            assert detector.detect(dummy_frame) == []
