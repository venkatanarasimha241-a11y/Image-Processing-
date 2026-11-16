"""
test_camera.py
--------------
Unit tests for RTSP Camera Engine and Camera Manager (rtsp_camera.py, camera.py, camera_manager.py).
Mocks cv2.VideoCapture to avoid needing a physical CCTV camera or live network RTSP feed.
"""

import time
import numpy as np
import pytest
import cv2
from unittest.mock import MagicMock, patch

from rtsp_camera import RTSPCamera, mask_rtsp_url
from camera_manager import CameraManager, TrackedEntity


class TestRTSPCamera:

    def test_01_successful_camera_initialization(self):
        """Scenario 1: Successful camera initialization."""
        cam = RTSPCamera(
            camera_id="cam_main",
            source="rtsp://admin:pass@192.168.1.50:554/ch1",
            name="Main Gate",
            location="Gate 1"
        )
        assert cam.camera_id == "cam_main"
        assert cam.name == "Main Gate"
        assert cam.location == "Gate 1"
        assert cam.is_running is False
        assert cam.connection_status == "INITIALIZING"

    def test_02_invalid_rtsp_url_handling(self):
        """Scenario 2: Invalid RTSP URL handling in verify_stream."""
        # Empty string
        success, msg, info = RTSPCamera.verify_stream("")
        assert success is False
        assert "No camera source provided" in msg

        # None input
        success, msg, info = RTSPCamera.verify_stream(None)
        assert success is False

    def test_03_camera_connection_failure(self):
        """Scenario 3: Camera connection failure when VideoCapture fails to open."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cap_cls.return_value = mock_cap

            cam = RTSPCamera(camera_id="cam_fail", source="rtsp://invalid-ip:554/live")
            success = cam._open_source()
            assert success is False
            assert cam.connection_status == "RECONNECTING"
            assert "Could not open video capture device" in cam.last_error

    def test_04_frame_capture_success(self):
        """Scenario 4: Frame capture success returns valid numpy frame."""
        dummy_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, dummy_frame)
            mock_cap_cls.return_value = mock_cap

            cam = RTSPCamera(camera_id="cam_ok", source="0")
            success = cam._open_source()
            assert success is True
            assert cam.connection_status == "CONNECTED"

            ret, frame = cam.read_frame()
            assert ret is True
            assert frame is not None
            assert frame.shape == (480, 640, 3)

    def test_05_frame_capture_failure(self):
        """Scenario 5: Frame capture failure returns (False, None)."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            # Fails on read
            mock_cap.read.return_value = (False, None)
            mock_cap_cls.return_value = mock_cap

            cam = RTSPCamera(camera_id="cam_read_fail", source="0")
            success = cam._open_source()
            assert success is False
            ret, frame = cam.read_frame()
            assert ret is False
            assert frame is None

    def test_06_empty_or_invalid_frames(self):
        """Scenario 6: Handle empty or zero-size frames safely."""
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, empty_frame)
            mock_cap_cls.return_value = mock_cap

            cam = RTSPCamera(camera_id="cam_empty", source="0")
            success = cam._open_source()
            assert success is False
            assert "failed to read initial frame" in cam.last_error.lower()

    def test_07_camera_disconnection_during_processing(self):
        """Scenario 7: Camera disconnection during grabber loop triggers reconnect status."""
        cam = RTSPCamera(camera_id="cam_disc", source="rtsp://fake:554/stream")
        cam.is_running = True
        cam.connection_status = "CONNECTED"

        # Simulate disconnection
        cam.disconnect()
        assert not cam.is_running
        assert cam.connection_status == "DISCONNECTED"
        ret, frame = cam.read_frame()
        assert ret is False
        assert frame is None

    def test_08_reconnection_logic(self):
        """Scenario 8: Reconnection worker initiates retries upon stream loss."""
        cam = RTSPCamera(camera_id="cam_reconn", source="0")
        cam.is_running = True
        cam.connection_status = "RECONNECTING"

        # Mock _open_source to succeed on reconnect attempt
        with patch.object(cam, "_open_source", return_value=True) as mock_open:
            cam._trigger_reconnect()
            assert cam._reconnect_thread is not None
            cam._reconnect_thread.join(timeout=2.0)
            mock_open.assert_called()

    def test_09_correct_release_of_videocapture_resources(self):
        """Scenario 9: VideoCapture resources are properly released on stop/disconnect."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.return_value = (True, np.full((100, 100, 3), 50, dtype=np.uint8))
            mock_cap_cls.return_value = mock_cap

            cam = RTSPCamera(camera_id="cam_rel", source="0")
            cam._open_source()
            assert cam.cap is not None

            cam.stop()
            mock_cap.release.assert_called()
            assert cam.cap is None

    def test_10_exceptions_during_frame_processing(self):
        """Scenario 10: Exceptions during capture do not crash the camera."""
        with patch("cv2.VideoCapture") as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.read.side_effect = Exception("OpenCV decoding driver fault")
            mock_cap_cls.return_value = mock_cap

            cam = RTSPCamera(camera_id="cam_err", source="0")
            success = cam._open_source()
            assert success is False
            assert "OpenCV decoding driver fault" in cam.last_error

    def test_11_graceful_shutdown_of_threads(self):
        """Scenario 11: Graceful shutdown of grabber thread on stop()."""
        cam = RTSPCamera(camera_id="cam_shut", source="0")
        with patch.object(cam, "_open_source", return_value=True):
            cam.start()
            assert cam.is_running is True
            assert cam._grabber_thread is not None
            assert cam._grabber_thread.is_alive()

            cam.stop()
            assert not cam.is_running
            cam._grabber_thread.join(timeout=2.0)
            assert not cam._grabber_thread.is_alive()

    def test_12_prevention_of_crashes_when_frames_unavailable(self):
        """Scenario 12: Calling read_frame when camera is offline returns (False, None) safely."""
        cam = RTSPCamera(camera_id="cam_offline")
        cam.connection_status = "OFFLINE"
        ret, frame = cam.read_frame()
        assert ret is False
        assert frame is None

        # Credential masking utility test
        masked = mask_rtsp_url("rtsp://admin:superSecretPass123@192.168.1.100:554/live")
        assert "superSecretPass123" not in masked
        assert "******" in masked

    def test_13_camera_wrapper_module_exports(self):
        """Scenario 13: Verify camera.py backward-compatible wrapper exports."""
        import camera
        assert hasattr(camera, "CameraManager")
        assert hasattr(camera, "RTSPCamera")
        assert hasattr(camera, "TrackedEntity")
        assert hasattr(camera, "generate_mjpeg_stream")
        assert hasattr(camera, "mask_rtsp_url")
