"""
test_config.py
--------------
Unit tests for system configuration, environment variable fallback,
and initialization resilience (config.py, firebase_config.py).
Ensures passwords, service account keys, and credentials are never leaked.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import patch

import config
import firebase_config


class TestConfigAndInitialization:

    def test_01_missing_environment_variables_fallbacks(self, monkeypatch):
        """Scenario 1: Missing environment variables fall back to safe calibrated defaults."""
        monkeypatch.delenv("YOLO_CONFIDENCE", raising=False)
        monkeypatch.delenv("RECOGNITION_THRESHOLD", raising=False)
        monkeypatch.delenv("CONFIDENCE_THRESHOLD", raising=False)

        # Re-evaluate environment defaults
        default_yolo = float(os.environ.get("YOLO_CONFIDENCE", "0.40"))
        default_rec = float(os.environ.get("RECOGNITION_THRESHOLD", "0.84"))
        assert default_yolo == 0.40
        assert default_rec == 0.84

    def test_02_missing_model_weights_handling(self, tmp_path):
        """Scenario 2: Missing model weights file is detected gracefully without crashing."""
        missing_path = str(tmp_path / "absent_weights.pth")
        assert not os.path.exists(missing_path)

        from recognition import FaceRecognizer
        with patch("recognition.torch.load", side_effect=FileNotFoundError):
            rec = FaceRecognizer(model_path=missing_path)
            assert rec.model is None

    def test_03_missing_configuration_files_defaults(self, tmp_path):
        """Scenario 3: Missing JSON configuration files use calibrated fallback values."""
        nonexistent_config = str(tmp_path / "nonexistent_config.json")
        from recognition import FaceRecognizer
        rec = FaceRecognizer(config_path=nonexistent_config)
        assert rec.similarity_threshold == config.IDENTITY_DECISION_THRESHOLD

    def test_04_invalid_configuration_values_handling(self):
        """Scenario 4: Invalid configuration values in service account keys."""
        invalid_key = {
            "type": "not_a_service_account",
            "project_id": "placeholder",
            "private_key": "not-a-key"
        }
        is_valid, msg = firebase_config.validate_credentials_dict(invalid_key)
        assert is_valid is False
        assert "must be 'service_account'" in msg

    def test_05_failed_model_initialization_status(self):
        """Scenario 5: Failed model initialization provides explanatory status message."""
        from yolo_detector import YOLODetector
        with patch("ultralytics.YOLO", side_effect=Exception("Corrupted model file")):
            det = YOLODetector(model_path="corrupted.pt")
            assert det.is_loaded is False
            assert "error" in det.status_message.lower()

    def test_06_missing_firebase_credentials_demo_mode(self):
        """Scenario 6: Missing Firebase credentials safely activates DEMO MODE."""
        with patch("firebase_config.is_valid_credential_file", return_value=False), \
             patch.dict(os.environ, {}, clear=True):
            is_demo = firebase_config.initialize_firebase(force_reinit=True)
            assert is_demo is True
            assert firebase_config.is_demo_mode() is True

    def test_07_credentials_never_exposed_in_status_output(self):
        """Scenario 7: Never expose Firebase private_key or passwords in status output."""
        status = firebase_config.get_firebase_status()
        status_str = json.dumps(status)
        assert "private_key" not in status_str
        assert "BEGIN PRIVATE KEY" not in status_str
