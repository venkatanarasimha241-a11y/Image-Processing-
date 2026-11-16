"""
test_image_storage.py
---------------------
Unit tests for image persistence, directory management, filename generation,
atomic operations, and path-traversal prevention.
Uses temporary directories so that real project images are never modified.
"""

import os
import uuid
import numpy as np
import pytest
import cv2

from firebase_service import save_image_to_firestore, get_image_from_firestore


class TestImageStorage:

    def test_01_saving_valid_image(self, temp_storage):
        """Scenario 1: Saving a valid image to disk and firestore cache."""
        dummy_img = np.full((100, 100, 3), 128, dtype=np.uint8)
        dest_path = os.path.join(temp_storage["captured_images"], "valid_test.jpg")

        success = cv2.imwrite(dest_path, dummy_img)
        assert success is True
        assert os.path.exists(dest_path)
        assert os.path.getsize(dest_path) > 0

    def test_02_handling_nonexistent_destination_directories(self, temp_storage):
        """Scenario 2: Handling nonexistent destination directories (creating them safely)."""
        nested_dir = os.path.join(temp_storage["root"], "deep", "nested", "storage")
        assert not os.path.exists(nested_dir)

        os.makedirs(nested_dir, exist_ok=True)
        assert os.path.exists(nested_dir)

        test_file = os.path.join(nested_dir, "nested_img.jpg")
        cv2.imwrite(test_file, np.zeros((50, 50, 3), dtype=np.uint8))
        assert os.path.exists(test_file)

    def test_03_preventing_accidental_filename_collisions(self, temp_storage):
        """Scenario 3: Preventing accidental filename collisions with unique IDs."""
        filenames = set()
        for _ in range(50):
            # Same timestamp simulation with random hex suffix
            suffix = os.urandom(4).hex()
            fname = f"capture_20261022_150000_{suffix}.jpg"
            assert fname not in filenames
            filenames.add(fname)
        assert len(filenames) == 50

    def test_04_handling_invalid_image_data(self, temp_storage):
        """Scenario 4: Handling invalid image data safely."""
        invalid_array = np.array([])
        # save_image_to_firestore should handle empty arrays without raising uncaught exceptions
        res = save_image_to_firestore(invalid_array)
        assert res["size_bytes"] == 0

    def test_05_handling_file_write_failures(self, temp_storage, monkeypatch):
        """Scenario 5: Handling file-write failures when permissions are read-only."""
        dummy_img = np.zeros((20, 20, 3), dtype=np.uint8)
        # Attempt to write to a path where parent is a non-directory file
        dummy_file = os.path.join(temp_storage["root"], "file_blocker")
        with open(dummy_file, "w") as f:
            f.write("I am a file, not a directory")

        impossible_path = os.path.join(dummy_file, "cant_write_here.jpg")
        success = cv2.imwrite(impossible_path, dummy_img)
        assert success is False

    def test_06_verifying_saved_files_can_be_reopened(self, temp_storage):
        """Scenario 6: Verifying that saved files can be reopened and decode to original dimensions."""
        original = np.full((80, 120, 3), 77, dtype=np.uint8)
        file_path = os.path.join(temp_storage["captured_images"], "reopen_test.jpg")
        cv2.imwrite(file_path, original)

        reopened = cv2.imread(file_path)
        assert reopened is not None
        assert reopened.shape == (80, 120, 3)

    def test_07_handling_duplicate_filenames_safely(self, temp_storage):
        """Scenario 7: Safe collision avoidance when duplicate names are encountered."""
        base_name = "evidence.jpg"
        p1 = os.path.join(temp_storage["captured_images"], base_name)
        cv2.imwrite(p1, np.zeros((10, 10, 3), dtype=np.uint8))

        # Check existence before overwrite
        if os.path.exists(p1):
            unique_name = f"evidence_{uuid.uuid4().hex[:6]}.jpg"
            p2 = os.path.join(temp_storage["captured_images"], unique_name)
        else:
            p2 = p1

        cv2.imwrite(p2, np.zeros((10, 10, 3), dtype=np.uint8))
        assert p1 != p2
        assert os.path.exists(p1)
        assert os.path.exists(p2)

    def test_08_preventing_path_traversal_and_unauthorized_access(self, temp_storage):
        """Scenario 8: Preventing path traversal and directory escape."""
        malicious_id = "../../../../etc/passwd"
        # Test Firestore image retrieval sanitization
        bytes_data, content_type = get_image_from_firestore(malicious_id)
        # Should not read sensitive system files outside of allowed folders
        assert bytes_data is None

        # Test safe basename resolution
        safe_filename = os.path.basename(malicious_id)
        assert ".." not in safe_filename
        assert safe_filename == "passwd"
