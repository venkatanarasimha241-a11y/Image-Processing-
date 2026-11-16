"""
conftest.py
-----------
Shared pytest fixtures and test doubles for unit testing the AI Camera Surveillance System.
Provides isolated environments, mocked neural network models, synthetic image generators,
and mock database services.
"""

import os
import sys
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
import cv2
import torch

# Ensure workspace root is in python path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)


@pytest.fixture(autouse=True)
def prevent_real_camera_access(monkeypatch):
    """
    Safety fixture: ensures tests do not attempt to open real physical cameras or network sockets.
    """
    monkeypatch.setenv("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")


@pytest.fixture
def sample_embedding():
    """Generates a valid 128-dimensional L2-normalized face embedding."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(128).astype(np.float32)
    norm = np.linalg.norm(vec)
    return vec / norm


@pytest.fixture
def sample_face_bgr():
    """Generates a synthetic 112x112 BGR face crop with facial contrast."""
    img = np.full((112, 112, 3), 180, dtype=np.uint8)
    # Simulated eyes
    cv2.circle(img, (35, 45), 8, (50, 50, 50), -1)
    cv2.circle(img, (75, 45), 8, (50, 50, 50), -1)
    # Simulated nose & mouth
    cv2.line(img, (55, 52), (55, 68), (70, 70, 70), 2)
    cv2.ellipse(img, (55, 82), (20, 10), 0, 0, 180, (40, 40, 150), 2)
    return img


@pytest.fixture
def sample_portrait_bgr():
    """Generates a 240x240 portrait photo with a detectable centered face."""
    img = np.full((240, 240, 3), 210, dtype=np.uint8)
    # Head contour
    cv2.ellipse(img, (120, 120), (60, 75), 0, 0, 360, (190, 160, 140), -1)
    # Eyes
    cv2.circle(img, (95, 105), 8, (40, 30, 30), -1)
    cv2.circle(img, (145, 105), 8, (40, 30, 30), -1)
    # Nose & mouth
    cv2.line(img, (120, 110), (120, 135), (100, 80, 80), 2)
    cv2.ellipse(img, (120, 150), (25, 12), 0, 0, 180, (40, 40, 120), 2)
    return img


@pytest.fixture
def sample_multi_face_bgr():
    """Generates a photo containing two distinct faces."""
    img = np.full((240, 400, 3), 200, dtype=np.uint8)
    # Person 1
    cv2.ellipse(img, (100, 120), (50, 65), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (80, 110), 6, (30, 30, 30), -1)
    cv2.circle(img, (120, 110), 6, (30, 30, 30), -1)
    # Person 2
    cv2.ellipse(img, (300, 120), (50, 65), 0, 0, 360, (180, 150, 130), -1)
    cv2.circle(img, (280, 110), 6, (30, 30, 30), -1)
    cv2.circle(img, (320, 110), 6, (30, 30, 30), -1)
    return img


@pytest.fixture
def sample_blank_bgr():
    """Generates a blank 200x200 image with no faces."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def sample_blurry_face_bgr():
    """Generates a severely blurred face image."""
    img = np.full((100, 100, 3), 150, dtype=np.uint8)
    # Severe Gaussian blur
    return cv2.GaussianBlur(img, (45, 45), 0)


@pytest.fixture
def mock_cnn_model(sample_embedding):
    """Mocks FaceCNN PyTorch model returning valid embeddings."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    # extract_features returns a torch Tensor of shape (1, 128)
    emb_tensor = torch.from_numpy(sample_embedding).unsqueeze(0)
    mock_model.extract_features.return_value = emb_tensor
    mock_model.return_value = torch.zeros((1, 6))
    return mock_model


@pytest.fixture
def temp_storage(tmp_path, monkeypatch):
    """Configures isolated temporary directories for file storage tests."""
    known_dir = tmp_path / "known_faces"
    captured_dir = tmp_path / "captured_images"
    models_dir = tmp_path / "models"
    known_dir.mkdir(parents=True, exist_ok=True)
    captured_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("config.KNOWN_FACES_DIR", str(known_dir))
    monkeypatch.setattr("config.CAPTURED_IMAGES_DIR", str(captured_dir))
    monkeypatch.setattr("config.MODELS_DIR", str(models_dir))

    return {
        "root": tmp_path,
        "known_faces": str(known_dir),
        "captured_images": str(captured_dir),
        "models": str(models_dir)
    }


@pytest.fixture
def test_client():
    """Provides Flask test client with clean configuration."""
    from app import app
    app.config["TESTING"] = True
    app.config["DEBUG"] = False
    with app.test_client() as client:
        yield client
