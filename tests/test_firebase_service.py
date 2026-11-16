"""
test_firebase_service.py
------------------------
Unit tests for Firebase Firestore Service (firebase_service.py) and Firebase Config (firebase_config.py).
Mocks Firestore client to ensure no test data is ever written to production databases.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest

import firebase_service
import firebase_config


@pytest.fixture
def clean_local_store():
    """Backs up and resets in-memory stores for clean testing."""
    orig_persons = list(firebase_service._local_persons)
    orig_detections = list(firebase_service._local_detections)
    yield
    firebase_service._local_persons = orig_persons
    firebase_service._local_detections = orig_detections


class TestFirebaseService:

    def test_01_successful_person_record_creation(self, clean_local_store):
        """Scenario 1: Successful person-record creation."""
        person_data = {
            "person_id": "TEST_STU001",
            "name": "Arun Kumar",
            "department": "Computer Science & Engineering",
            "role": "Student",
            "class": "CSE-3B",
            "year": "3rd Year"
        }
        res = firebase_service.add_person(person_data)
        assert res["person_id"] == "TEST_STU001"
        assert res["name"] == "Arun Kumar"
        assert res["status"] == "ACTIVE"
        # Check retrieval
        retrieved = firebase_service.get_person_by_id("TEST_STU001")
        assert retrieved is not None
        assert retrieved["name"] == "Arun Kumar"

    def test_02_successful_retrieval_of_person_record(self, clean_local_store):
        """Scenario 2: Successful retrieval of a person record."""
        # Using pre-seeded record
        person = firebase_service.get_person_by_id("STU001")
        assert person is not None
        assert person["name"] == "Rahul Kumar"
        assert person["role"] == "Student"

        # Check detail formatting
        details = firebase_service.get_person_details("STU001")
        assert details["status"] == "known"
        assert details["person_type"] == "Student"
        assert "timetable" in details

    def test_03_updating_existing_record(self, clean_local_store):
        """Scenario 3: Updating an existing person record."""
        updated = firebase_service.update_person("STU001", {"cgpa": "9.50", "class": "CSE-4A"})
        assert updated is not None
        assert updated["cgpa"] == "9.50"
        assert updated["class"] == "CSE-4A"

        fetched = firebase_service.get_person_by_id("STU001")
        assert fetched is not None
        assert fetched["cgpa"] == "9.50"

    def test_04_handling_missing_record(self):
        """Scenario 4: Handling a missing person record returns None or Unknown formatted record."""
        # By ID
        res = firebase_service.get_person_by_id("NON_EXISTENT_ID_999")
        assert res is None

        # By Details
        details = firebase_service.get_person_details("NON_EXISTENT_ID_999")
        assert details["status"] == "unknown"
        assert details["person_type"] == "Unknown"
        assert "alert" in details

    def test_05_duplicate_registration_handling(self, clean_local_store):
        """Scenario 5: Duplicate registration prevention / checking."""
        existing = firebase_service.get_person_by_id("STU001")
        assert existing is not None
        # System can detect existing ID
        is_duplicate = (existing.get("status") != "unknown")
        assert is_duplicate is True

    def test_06_missing_or_invalid_input_fields(self, clean_local_store):
        """Scenario 6: Handle missing optional fields with sensible defaults."""
        minimal_data = {"name": "Test Minimal"}
        res = firebase_service.add_person(minimal_data)
        assert res["person_id"].startswith("STU") or res["person_id"].startswith("FAC")
        assert res["status"] == "ACTIVE"
        assert res["image_url"] == "/static/images/default_avatar.svg"

    def test_07_firebase_operation_failures_fallback_to_local(self, clean_local_store):
        """Scenario 7: Operation failures in Firestore fall back safely to local store."""
        with patch("firebase_service.get_firestore_client") as mock_db_getter, \
             patch("firebase_service.is_demo_mode", return_value=False):
            mock_db = MagicMock()
            mock_db.collection.side_effect = Exception("Firestore Quota Exceeded / Unavailable")
            mock_db_getter.return_value = mock_db

            # add_person should not crash; it logs error and persists in local memory
            res = firebase_service.add_person({"person_id": "FALLBACK_01", "name": "Fallback User"})
            assert res["person_id"] == "FALLBACK_01"

    def test_08_database_connection_initialization_failures(self):
        """Scenario 8: Database initialization failure enters DEMO MODE."""
        with patch("firebase_config.is_valid_credential_file", return_value=False):
            is_demo = firebase_config.initialize_firebase(force_reinit=True)
            assert is_demo is True
            assert firebase_config.is_demo_mode() is True

    def test_09_correct_handling_of_empty_query_results(self):
        """Scenario 9: Correct handling of empty query results."""
        # Query with non-matching filter
        results = firebase_service.get_detections(filters={"camera": "NonExistentCam999"})
        assert isinstance(results, list)
        assert len(results) == 0

    def test_10_serialization_of_timestamps_and_supported_types(self, clean_local_store):
        """Scenario 10: Serialization of timestamps and data types."""
        det_data = {
            "camera_id": "gate_1",
            "person_id": "STU001",
            "person_name": "Rahul Kumar",
            "confidence": 0.95,
            "recognition_status": "known"
        }
        saved = firebase_service.save_detection(det_data)
        assert "id" in saved
        assert "timestamp" in saved
        assert "raw_timestamp" in saved
        # Validate ISO format
        dt = datetime.fromisoformat(saved["raw_timestamp"])
        assert dt is not None

    def test_11_detection_record_creation(self, clean_local_store):
        """Scenario 11: Detection-record creation and retrieval."""
        initial_count = len(firebase_service.get_detections())
        det_data = {
            "camera_id": "gate_1",
            "person_id": "UNKNOWN",
            "recognition_status": "unknown",
            "confidence": 0.42
        }
        saved = firebase_service.save_detection(det_data)
        assert saved["id"] is not None

        new_detections = firebase_service.get_detections()
        assert len(new_detections) == initial_count + 1
        assert new_detections[0]["id"] == saved["id"]

    def test_12_association_between_detection_and_registered_person(self, clean_local_store):
        """Scenario 12: Correct association between detection and registered person increments count."""
        person = firebase_service.get_person_by_id("STU001")
        assert person is not None
        initial_detections = person.get("detections_count", 0)

        det_data = {
            "camera_id": "cctv_entrance",
            "person_id": "STU001",
            "person_name": "Rahul Kumar",
            "recognition_status": "known"
        }
        firebase_service.save_detection(det_data)

        updated_person = firebase_service.get_person_by_id("STU001")
        assert updated_person is not None
        assert updated_person["detections_count"] == initial_detections + 1
        assert updated_person["last_detected"] == "Just now"
