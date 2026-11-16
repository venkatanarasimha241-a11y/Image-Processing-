"""
test_cross_module.py
--------------------
Cross-module interaction and interface verification tests.
Tests the end-to-end integration boundaries between:
- Image Upload / Camera Grabber -> YOLO Detection -> Face Preprocessing -> CNN Embedding -> Identity Matching
- Recognition Event -> Firestore Detection Record / Storage Sync
- Failure resilience and graceful degradation across module boundaries
"""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import cv2

from camera_manager import CameraManager
from rtsp_camera import RTSPCamera
from recognition import FaceRecognizer
from yolo_detector import YOLODetector
import firebase_service


@pytest.fixture
def clean_system(temp_storage, mock_cnn_model, sample_embedding):
    """Initializes integrated test system with mocked deep learning inference."""
    with patch("recognition.torch.load") as mock_torch_load, \
         patch("recognition.FaceCNN") as mock_cnn_cls, \
         patch("ultralytics.YOLO") as mock_yolo_cls:

        mock_torch_load.return_value = {
            "num_classes": 6,
            "architecture": "FaceCNN_4Block",
            "model_state_dict": {}
        }
        mock_cnn_cls.return_value = mock_cnn_model

        mock_yolo_model = MagicMock()
        mock_yolo_model.names = {0: "person", 67: "cell phone"}
        mock_yolo_cls.return_value = mock_yolo_model

        rec = FaceRecognizer(
            model_path=os.path.join(temp_storage["models"], "model.pth"),
            class_names_path=os.path.join(temp_storage["root"], "classes.json"),
            metadata_path=os.path.join(temp_storage["root"], "meta.json")
        )
        rec.model = mock_cnn_model

        # Enroll a test student
        rec.enrolled_gallery["STU001"] = {
            "person_id": "STU001",
            "name": "Rahul Kumar",
            "embedding": sample_embedding,
            "department": "CSE",
            "role": "Student"
        }
        rec.similarity_threshold = 0.84

        yolo = YOLODetector()
        return rec, yolo, mock_yolo_model


class TestCrossModuleInteractions:

    def test_01_upload_to_embedding_to_matching_pipeline(self, clean_system, sample_portrait_bgr, sample_embedding):
        """Cross-Module 1: Uploaded image -> face preprocessing -> CNN embedding -> identity matching."""
        recognizer, yolo, mock_yolo = clean_system
        h, w = sample_portrait_bgr.shape[:2]

        # Process portrait image
        res = recognizer.recognize(
            frame=sample_portrait_bgr,
            person_bbox=[0, 0, w, h],
            track_id=1,
            is_single_image=True
        )
        assert res is not None
        assert res["face_detected"] is True
        assert res["status"] == "known"
        assert res["person_id"] == "STU001"
        assert res["similarity_score"] >= 0.84

    def test_02_cctv_frame_to_yolo_to_recognition_pipeline(self, clean_system, sample_portrait_bgr, sample_embedding):
        """Cross-Module 2: CCTV frame -> YOLO person detection -> face processing -> CNN embedding -> identity matching."""
        recognizer, yolo, mock_yolo = clean_system
        h, w = sample_portrait_bgr.shape[:2]

        import torch
        from typing import Optional
        class MockBox:
            def __init__(self, cls_id: int, conf: float, xyxy: list, track_id: Optional[int] = None):
                self.cls = torch.tensor([cls_id])
                self.conf = torch.tensor([conf])
                self.xyxy = torch.tensor([xyxy])
                self.id = torch.tensor([track_id]) if track_id is not None else None

        mock_res = MagicMock()
        mock_res.boxes = [MockBox(0, 0.92, [10, 10, w - 10, h - 10], track_id=5)]
        mock_yolo.track.return_value = [mock_res]

        # Step 1: Detect with YOLO
        dets = yolo.detect(sample_portrait_bgr, is_single_image=False)
        assert len(dets) == 1
        assert dets[0]["is_person"] is True

        # Step 2: Pass detected person box to Face Recognizer
        rec_res = recognizer.recognize(
            frame=sample_portrait_bgr,
            person_bbox=dets[0]["bbox"],
            track_id=dets[0]["track_id"],
            yolo_conf=dets[0]["confidence"],
            is_single_image=True
        )
        assert rec_res["face_detected"] is True
        assert rec_res["person_id"] == "STU001"
        assert rec_res["status"] == "known"

    def test_03_recognized_person_to_firestore_record(self, clean_system):
        """Cross-Module 3: Recognized person creates structured Firestore detection record."""
        det_record = {
            "camera_id": "gate_camera_1",
            "person_id": "STU001",
            "person_name": "Rahul Kumar",
            "recognition_status": "known",
            "confidence": 0.95,
            "department": "Computer Science & Engineering",
            "role": "Student"
        }
        saved = firebase_service.save_detection(det_record)
        assert saved["id"] is not None
        assert saved["person_id"] == "STU001"

        # Check person telemetry updated
        person = firebase_service.get_person_by_id("STU001")
        assert person is not None
        assert person["last_detected"] == "Just now"

    def test_04_unknown_person_creates_evidence_record(self, clean_system, temp_storage):
        """Cross-Module 4: Unknown person saves evidence crop and logs alert detection record."""
        dummy_crop = np.full((120, 120, 3), 100, dtype=np.uint8)
        img_id = "test_unknown_event_01"

        saved_img = firebase_service.save_image_to_firestore(
            image_input=dummy_crop,
            image_id=img_id,
            filename="unknown_crop.jpg",
            metadata={"status": "unknown"}
        )
        assert saved_img["image_id"] == img_id
        assert saved_img["image_url"].startswith("/api/firestore/image/")

        unknown_detection = {
            "camera_id": "gate_camera_1",
            "person_id": "UNKNOWN",
            "person_name": "Unknown Person",
            "recognition_status": "unknown",
            "confidence": 0.40,
            "image_url": saved_img["image_url"],
            "image_id": img_id
        }
        saved_det = firebase_service.save_detection(unknown_detection)
        assert saved_det["recognition_status"] == "unknown"
        assert saved_det["image_id"] == img_id

    def test_05_camera_failure_error_handling(self):
        """Cross-Module 5: Camera stream failure sets RECONNECTING state without crashing system."""
        with patch("cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            cam = RTSPCamera(camera_id="cam_stream_fail", source="rtsp://unreachable:554/live")
            success = cam._open_source()
            assert success is False
            assert cam.connection_status == "RECONNECTING"
            ret, frame = cam.read_frame()
            assert ret is False
            assert frame is None

    def test_06_recognition_failure_safe_degradation(self, clean_system, sample_blank_bgr):
        """Cross-Module 6: Recognition failure degrades safely to 'no_face' without crashing."""
        recognizer, yolo, mock_yolo = clean_system
        # Force model error
        recognizer.model.extract_features.side_effect = RuntimeError("Feature extraction failed")

        h, w = sample_blank_bgr.shape[:2]
        res = recognizer.recognize(
            frame=sample_blank_bgr,
            person_bbox=[0, 0, w, h],
            track_id=1,
            is_single_image=True
        )
        assert res["status"] in ["no_face", "unknown"]
        assert res["face_detected"] is False
