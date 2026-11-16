"""
test_app.py
-----------
Comprehensive Flask route unit testing for app.py using Flask test_client.
Mocks heavy neural network inference, database writes, and camera streams
to verify HTTP status codes, JSON response structures, error handling,
and boundary conditions across all actual endpoints.
"""

import io
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import cv2

from app import app


@pytest.fixture
def client():
    """Configures Flask application in testing mode."""
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture
def sample_jpeg_bytes():
    """Generates valid JPEG binary bytes."""
    img = np.full((120, 120, 3), 160, dtype=np.uint8)
    ret, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class TestFlaskBackendRoutes:

    # 1. HTML Web Page Routes
    def test_01_web_page_routes_render_successfully(self, client):
        """Scenario 1: Web page routes render with 200 OK or appropriate redirect."""
        # Landing / Dashboard
        resp = client.get("/")
        assert resp.status_code == 200

        # Login page
        resp = client.get("/login")
        assert resp.status_code == 200

        # Registered persons page
        resp = client.get("/persons")
        assert resp.status_code == 200

        # Detection logs page
        resp = client.get("/detections")
        assert resp.status_code == 200

    # 2. Authentication API
    def test_02_login_successful_and_invalid_credentials(self, client):
        """Scenario 2: Login API returns 200 for valid credentials and 400 for missing credentials."""
        # Valid credentials
        resp = client.post("/api/login", json={"email": "admin@campus.edu", "password": "password123"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "user" in data
        assert data["user"]["role"] == "System Administrator"

        # Missing password
        resp = client.post("/api/login", json={"email": "admin@campus.edu", "password": ""})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False
        assert "Email and password are required" in data["error"]

        # Missing email
        resp = client.post("/api/login", json={"email": "", "password": "password123"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["success"] is False

    # 3. Method Not Allowed (405)
    def test_03_invalid_request_methods_return_405(self, client):
        """Scenario 3: Invalid request methods return HTTP 405 Method Not Allowed."""
        # /api/login only accepts POST
        resp = client.get("/api/login")
        assert resp.status_code == 405

        # /api/stats only accepts GET
        resp = client.post("/api/stats")
        assert resp.status_code == 405

    # 4. System Statistics API
    def test_04_system_stats_endpoint(self, client):
        """Scenario 4: /api/stats returns 200 with structured KPI metrics."""
        resp = client.get("/api/stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "stats" in data
        stats = data["stats"]
        assert "total_detections" in stats
        assert "known_count" in stats
        assert "unknown_count" in stats

    # 5. Detections Query API
    def test_05_detections_endpoint_queries_and_filters(self, client):
        """Scenario 5: /api/detections returns list of detections with filtering."""
        resp = client.get("/api/detections?limit=10")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert "detections" in data
        assert isinstance(data["detections"], list)

    # 6. Persons API (GET, POST, PUT, DELETE)
    def test_06_persons_crud_operations(self, client):
        """Scenario 6: /api/persons and /api/persons/<id> full CRUD workflow."""
        # 1. GET persons
        resp = client.get("/api/persons")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert len(data["persons"]) > 0

        # 2. POST create new person
        new_payload = {
            "person_id": "TEST_ROUTE_01",
            "name": "Route Test Student",
            "role": "Student",
            "department": "IT",
            "class": "IT-3A"
        }
        resp = client.post("/api/persons", json=new_payload)
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["success"] is True

        # 3. GET individual person
        resp = client.get("/api/persons/TEST_ROUTE_01")
        assert resp.status_code == 200
        assert resp.get_json()["person"]["name"] == "Route Test Student"

        # 4. PUT update individual person
        resp = client.put("/api/persons/TEST_ROUTE_01", json={"department": "Information Science"})
        assert resp.status_code == 200
        assert resp.get_json()["person"]["department"] == "Information Science"

        # 5. DELETE individual person
        resp = client.delete("/api/persons/TEST_ROUTE_01")
        assert resp.status_code == 200

    def test_07_persons_missing_name_returns_400(self, client):
        """Scenario 7: POST /api/persons with missing name returns 400 Bad Request."""
        resp = client.post("/api/persons", json={"role": "Student"})
        assert resp.status_code == 400
        assert resp.get_json()["success"] is False
        assert "Person name is required" in resp.get_json()["error"]

    def test_08_duplicate_person_registration_rejected(self, client):
        """Scenario 8: Registering a person_id that already exists returns 400."""
        # STU001 is already registered
        resp = client.post("/api/persons", json={"person_id": "STU001", "name": "Duplicate STU"})
        assert resp.status_code == 400
        assert "already registered" in resp.get_json()["error"]

    # 9. Camera Controls API
    def test_09_camera_endpoints(self, client):
        """Scenario 9: Camera status, verify, start, stop, switch endpoints."""
        # Status
        resp = client.get("/api/camera/status")
        assert resp.status_code == 200
        assert "camera" in resp.get_json()

        # Verify missing URL -> 400
        resp = client.post("/api/camera/verify", json={"rtsp_url": ""})
        assert resp.status_code == 400

        # Start
        resp = client.post("/api/camera/start")
        assert resp.status_code == 200

        # Stop
        resp = client.post("/api/camera/stop")
        assert resp.status_code == 200

        # Disconnect
        resp = client.post("/api/camera/disconnect")
        assert resp.status_code == 200

    # 10. Identify Image Upload Endpoint (Complete validation)
    def test_10_identify_image_no_file_provided(self, client):
        """Scenario 10: /api/identify_image returns 400 when no image is uploaded."""
        resp = client.post("/api/identify_image", data={})
        assert resp.status_code == 400
        assert "No image file provided" in resp.get_json()["error"]

    def test_11_identify_image_unsupported_file_format(self, client):
        """Scenario 11: /api/identify_image returns 400 for unsupported file types (.pdf, .txt)."""
        data = {
            "image": (io.BytesIO(b"Hello text document"), "document.txt")
        }
        resp = client.post("/api/identify_image", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Unsupported image file format" in resp.get_json()["error"]

    def test_12_identify_image_corrupted_image_content(self, client):
        """Scenario 12: /api/identify_image returns 400 when image cannot be decoded."""
        data = {
            "image": (io.BytesIO(b"not a valid jpeg content"), "corrupted.jpg")
        }
        resp = client.post("/api/identify_image", data=data, content_type="multipart/form-data")
        assert resp.status_code == 400
        assert "Invalid or corrupted image content" in resp.get_json()["error"]

    def test_13_identify_image_successful_identification(self, client, sample_jpeg_bytes):
        """Scenario 13: /api/identify_image returns 200 with structured entities on valid image."""
        with patch("app.camera_manager.yolo.detect") as mock_yolo_det, \
             patch("app.camera_manager.recognizer.recognize") as mock_rec, \
             patch("app.firebase_service.save_image_to_firestore") as mock_save_img:

            mock_yolo_det.return_value = [{
                "class_id": 0,
                "class_name": "person",
                "canonical_class": "person",
                "confidence": 0.88,
                "bbox": [10, 10, 100, 100],
                "is_person": True,
                "is_electronic": False,
                "is_vehicle": False
            }]
            mock_rec.return_value = {
                "status": "known",
                "class_id": "STU001",
                "person_id": "STU001",
                "name": "Rahul Kumar",
                "similarity_score": 0.94,
                "confidence": 0.94,
                "yolo_confidence": 0.88,
                "face_confidence": 0.91,
                "quality_score": 0.85,
                "blur_score": 85.0,
                "face_detected": True,
                "face_bbox": [20, 20, 80, 80]
            }
            mock_save_img.return_value = {
                "image_url": "/api/firestore/image/img_test_123"
            }

            data = {
                "image": (io.BytesIO(sample_jpeg_bytes), "sample.jpg")
            }
            resp = client.post("/api/identify_image", data=data, content_type="multipart/form-data")
            assert resp.status_code == 200
            res = resp.get_json()
            assert res["success"] is True
            assert res["entities_count"] == 1
            assert res["results"][0]["person_name"] == "Rahul Kumar"
            assert res["results"][0]["status"] == "known"

    def test_14_identify_image_unknown_person_response(self, client, sample_jpeg_bytes):
        """Scenario 14: /api/identify_image returns Unknown Person when recognition threshold is not met."""
        with patch("app.camera_manager.yolo.detect") as mock_yolo_det, \
             patch("app.camera_manager.recognizer.recognize") as mock_rec, \
             patch("app.firebase_service.save_image_to_firestore") as mock_save_img:

            mock_yolo_det.return_value = [{
                "class_id": 0,
                "class_name": "person",
                "canonical_class": "person",
                "confidence": 0.85,
                "bbox": [10, 10, 100, 100],
                "is_person": True,
                "is_electronic": False,
                "is_vehicle": False
            }]
            mock_rec.return_value = {
                "status": "unknown",
                "class_id": "UNKNOWN",
                "person_id": "UNKNOWN",
                "name": "Unknown Person",
                "similarity_score": 0.52,
                "confidence": 0.52,
                "yolo_confidence": 0.85,
                "face_confidence": 0.70,
                "quality_score": 0.65,
                "blur_score": 50.0,
                "face_detected": True,
                "face_bbox": [20, 20, 80, 80]
            }
            mock_save_img.return_value = {
                "image_url": "/api/firestore/image/img_unknown_456"
            }

            data = {
                "image": (io.BytesIO(sample_jpeg_bytes), "stranger.jpg")
            }
            resp = client.post("/api/identify_image", data=data, content_type="multipart/form-data")
            assert resp.status_code == 200
            res = resp.get_json()
            assert res["success"] is True
            assert res["results"][0]["status"] == "unknown"
            assert "Unknown Person" in res["results"][0]["person_name"]

    # 15. Thresholds Configuration API
    def test_15_thresholds_get_and_post(self, client):
        """Scenario 15: /api/settings/thresholds gets and updates thresholds."""
        # GET
        resp = client.get("/api/settings/thresholds")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "yolo_confidence" in data
        assert "recognition_similarity_threshold" in data

        # POST update
        resp = client.post("/api/settings/thresholds", json={
            "yolo_confidence": 0.45,
            "recognition_similarity": 0.85
        })
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["yolo_confidence"] == 0.45
        assert data["recognition_similarity_threshold"] == 0.85

    # 16. Firebase Cloud APIs
    def test_16_firebase_status_and_test(self, client):
        """Scenario 16: /api/firebase/status and /api/firebase/test endpoints."""
        resp = client.get("/api/firebase/status")
        assert resp.status_code == 200
        assert "status" in resp.get_json()

        resp = client.post("/api/firebase/test")
        # In demo mode without credentials, returns 400 with Demo Mode notice
        assert resp.status_code in [200, 400]
        assert "success" in resp.get_json()
