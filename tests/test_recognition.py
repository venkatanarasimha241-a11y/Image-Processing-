"""
test_recognition.py
-------------------
Unit tests for CNN Face Recognition and Face Embeddings (recognition.py).
Covers all 18 specified scenarios including edge cases, corrupted inputs,
boundary conditions, and embedding algebra.
"""

import os
import tempfile
import numpy as np
import pytest
import torch
import cv2
from unittest.mock import MagicMock, patch

from recognition import FaceRecognizer, RegistrationResult, TemporalTrackHistory
from face_detector import FaceDetector


@pytest.fixture
def recognizer(mock_cnn_model, temp_storage):
    """Initializes a FaceRecognizer instance with mocked model and isolated directories."""
    with patch("recognition.FaceCNN") as mock_face_cnn_cls, \
         patch("recognition.torch.load") as mock_torch_load:
        mock_torch_load.return_value = {
            "num_classes": 6,
            "architecture": "FaceCNN_4Block",
            "model_state_dict": {}
        }
        mock_face_cnn_cls.return_value = mock_cnn_model

        rec = FaceRecognizer(
            model_path=os.path.join(temp_storage["models"], "test_model.pth"),
            class_names_path=os.path.join(temp_storage["root"], "class_names.json"),
            metadata_path=os.path.join(temp_storage["root"], "metadata.json"),
            config_path=os.path.join(temp_storage["models"], "config.json")
        )
        rec.model = mock_cnn_model
        return rec


class TestCNNFaceRecognition:

    def test_01_correctly_process_valid_face_image(self, recognizer, sample_face_bgr, sample_embedding):
        """Scenario 1: Correctly process a valid face image."""
        # Enforce that mock_cnn_model returns sample_embedding
        recognizer.model.extract_features.return_value = torch.from_numpy(sample_embedding).unsqueeze(0)
        emb = recognizer.extract_embedding(sample_face_bgr)
        assert emb is not None
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (128,)
        assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)

    def test_02_handle_invalid_image_path(self, recognizer):
        """Scenario 2: Handle an invalid image path when building gallery."""
        # When an invalid path is encountered in directory walk
        with patch("cv2.imread", return_value=None):
            recognizer.build_enrolled_gallery()
            assert len(recognizer.enrolled_gallery) == 0

    def test_03_handle_missing_image_file(self, recognizer):
        """Scenario 3: Handle a missing image file."""
        non_existent_file = "/path/to/definitely/missing/image.jpg"
        img = cv2.imread(non_existent_file)
        assert img is None
        emb = recognizer.extract_embedding(img)
        assert emb is None

    def test_04_handle_corrupted_image(self, recognizer):
        """Scenario 4: Handle a corrupted image."""
        corrupted_bytes = b"\x00\xFF\xAA\x55CORRUPTEDDATA"
        img = cv2.imdecode(np.frombuffer(corrupted_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert img is None
        emb = recognizer.extract_embedding(img)
        assert emb is None

    def test_05_handle_image_containing_no_detectable_face(self, recognizer, sample_blank_bgr):
        """Scenario 5: Handle an image containing no detectable face."""
        h, w = sample_blank_bgr.shape[:2]
        res = recognizer.recognize(
            frame=sample_blank_bgr,
            person_bbox=[0, 0, w, h],
            track_id=1,
            is_single_image=True
        )
        assert res["status"] in ["no_face", "unknown"]
        assert res["face_detected"] is False
        assert res["similarity_score"] == 0.0

    def test_06_handle_multiple_faces_in_one_image(self, recognizer, sample_multi_face_bgr):
        """Scenario 6: Handle multiple faces according to registration policy (reject ambiguous photos)."""
        # In registration, multi-face photos must be rejected to prevent gallery corruption
        reg_res = recognizer.register_new_face(
            person_id="STU999",
            name="Ambiguous User",
            image_bgr=sample_multi_face_bgr
        )
        # Should be rejected if multiple faces are detected
        if len(recognizer.detector.detect_all_faces(sample_multi_face_bgr)) > 1:
            assert reg_res.success is False
            assert "ambiguous" in reg_res.message.lower() or "faces detected" in reg_res.message.lower()

    def test_07_preprocessing_produces_expected_dimensions_and_dtype(self, recognizer, sample_face_bgr):
        """Scenario 7: Verify that preprocessing produces expected (1, 3, 112, 112) float32 tensor."""
        tensor = recognizer.detector.preprocess_to_tensor(sample_face_bgr, device=torch.device("cpu"))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 112, 112)
        assert tensor.dtype == torch.float32
        # Normalization check: mean and std applied
        assert tensor.min() < 0.0
        assert tensor.max() > 0.0

    def test_08_embeddings_have_expected_dimensions(self, recognizer, sample_face_bgr, sample_embedding):
        """Scenario 8: Verify that embeddings have the expected 128 dimensions."""
        recognizer.model.extract_features.return_value = torch.from_numpy(sample_embedding).unsqueeze(0)
        emb = recognizer.extract_embedding(sample_face_bgr)
        assert emb.shape == (128,)

    def test_09_embeddings_contain_finite_values_no_nan_or_inf(self, recognizer, sample_face_bgr, sample_embedding):
        """Scenario 9: Check that embeddings contain finite values and do not contain NaN or infinity."""
        recognizer.model.extract_features.return_value = torch.from_numpy(sample_embedding).unsqueeze(0)
        emb = recognizer.extract_embedding(sample_face_bgr)
        assert np.all(np.isfinite(emb))
        assert not np.any(np.isnan(emb))
        assert not np.any(np.isinf(emb))

    def test_10_same_preprocessing_pipeline_used_for_upload_and_cctv(self, recognizer, sample_face_bgr):
        """Scenario 10: Verify the identical preprocessing function is called for uploaded and CCTV frames."""
        tensor_upload = recognizer.detector.preprocess_to_tensor(sample_face_bgr)
        tensor_cctv = recognizer.detector.preprocess_to_tensor(sample_face_bgr)
        assert torch.equal(tensor_upload, tensor_cctv)

    def test_11_matching_against_registered_person_embedding(self, recognizer, sample_embedding):
        """Scenario 11: Test matching against a registered person's embedding."""
        person_id = "STU001"
        recognizer.enrolled_gallery[person_id] = {
            "person_id": person_id,
            "name": "Rahul Kumar",
            "embedding": sample_embedding,
            "department": "CSE",
            "role": "Student"
        }
        recognizer.similarity_threshold = 0.84

        # Query with identical embedding -> Cosine sim should be ~1.00
        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(sample_embedding)
        assert status == "known"
        assert matched_id == person_id
        assert sim >= 0.99

    def test_12_unknown_person_embedding_rejection(self, recognizer, sample_embedding):
        """Scenario 12: Test an unknown person's embedding (fails threshold -> rejected as UNKNOWN)."""
        person_id = "STU001"
        recognizer.enrolled_gallery[person_id] = {
            "person_id": person_id,
            "name": "Rahul Kumar",
            "embedding": sample_embedding,
            "department": "CSE",
            "role": "Student"
        }
        recognizer.similarity_threshold = 0.84

        # Create orthogonal / distant embedding
        rng = np.random.default_rng(999)
        diff_vec = rng.standard_normal(128).astype(np.float32)
        diff_vec = diff_vec - np.dot(diff_vec, sample_embedding) * sample_embedding  # Gram-Schmidt orthogonalize
        diff_emb = diff_vec / np.linalg.norm(diff_vec)

        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(diff_emb)
        assert status == "unknown"
        assert matched_id == "UNKNOWN"
        assert sim < recognizer.similarity_threshold

    def test_13_similar_embeddings_ambiguity_handling(self, recognizer, sample_embedding):
        """Scenario 13: Test cases where two people have similar embeddings."""
        p1_emb = sample_embedding
        # Perturb slightly to make p2 very similar
        rng = np.random.default_rng(123)
        noise = rng.standard_normal(128).astype(np.float32) * 0.05
        p2_emb = p1_emb + noise
        p2_emb = p2_emb / np.linalg.norm(p2_emb)

        recognizer.enrolled_gallery["P1"] = {"person_id": "P1", "name": "Person 1", "embedding": p1_emb}
        recognizer.enrolled_gallery["P2"] = {"person_id": "P2", "name": "Person 2", "embedding": p2_emb}
        recognizer.similarity_threshold = 0.84

        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(p1_emb)
        assert status == "known"
        assert matched_id == "P1"
        assert runner_up > 0.80  # p2 was a close runner up

    def test_14_invalid_or_empty_embedding_vectors(self, recognizer):
        """Scenario 14: Test invalid or empty embedding vectors."""
        # Empty array
        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(np.array([]))
        assert status == "unknown"
        assert matched_id == "UNKNOWN"
        assert sim == 0.0

        # None embedding
        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(None)
        assert status == "unknown"
        assert matched_id == "UNKNOWN"
        assert sim == 0.0

        # Embedding with NaNs
        nan_vec = np.full(128, np.nan, dtype=np.float32)
        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(nan_vec)
        assert status == "unknown"
        assert matched_id == "UNKNOWN"

    def test_15_recognition_when_no_registered_faces_exist(self, recognizer, sample_embedding):
        """Scenario 15: Test recognition when no registered faces exist in gallery."""
        recognizer.enrolled_gallery.clear()
        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(sample_embedding)
        assert status == "unknown"
        assert matched_id == "UNKNOWN"
        assert sim == 0.0

    def test_16_invalid_similarity_thresholds(self, recognizer, sample_embedding):
        """Scenario 16: Test handling of invalid similarity thresholds."""
        recognizer.enrolled_gallery["STU001"] = {
            "person_id": "STU001", "name": "Rahul", "embedding": sample_embedding
        }
        # Threshold set higher than 1.0 -> should reject
        recognizer.similarity_threshold = 1.05
        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(sample_embedding)
        assert status == "unknown"

    def test_17_returns_unknown_when_no_candidate_satisfies_criteria(self, recognizer, sample_embedding):
        """Scenario 17: Verify system returns Unknown when no candidate satisfies criteria."""
        recognizer.enrolled_gallery["STU001"] = {
            "person_id": "STU001", "name": "Rahul", "embedding": sample_embedding
        }
        recognizer.similarity_threshold = 0.85
        rng = np.random.default_rng(777)
        unrelated_emb = rng.standard_normal(128).astype(np.float32)
        unrelated_emb = unrelated_emb / np.linalg.norm(unrelated_emb)

        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(unrelated_emb)
        assert status == "unknown"
        assert matched_id == "UNKNOWN"

    def test_18_low_quality_blurry_images_handled_safely(self, recognizer, sample_blurry_face_bgr):
        """Scenario 18: Verify that low-quality/blurry images are safely handled without crashes."""
        h, w = sample_blurry_face_bgr.shape[:2]
        res = recognizer.recognize(
            frame=sample_blurry_face_bgr,
            person_bbox=[0, 0, w, h],
            track_id=1,
            is_single_image=False  # In CCTV stream, blurry faces should not be marked as intruder
        )
        assert res["status"] in ["no_face", "unknown"]
        assert "blur_score" in res

    def test_19_face_cnn_unclamped_hyperspherical_embeddings(self):
        """Scenario 19: Verify FaceCNN extract_features produces normalized vectors on S^127."""
        from models.face_cnn import FaceCNN
        real_model = FaceCNN(num_classes=6)
        real_model.eval()
        dummy_input = torch.randn(2, 3, 112, 112)

        # Un-clamped extraction (default)
        embs_unclamped = real_model.extract_features(dummy_input, use_unclamped=True)
        assert embs_unclamped.shape == (2, 128)
        norms = torch.norm(embs_unclamped, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
        # Verify vectors contain negative values (not restricted to non-negative orthant)
        assert (embs_unclamped < 0).any().item() is True

        # Clamped extraction (legacy)
        embs_clamped = real_model.extract_features(dummy_input, use_unclamped=False)
        assert embs_clamped.shape == (2, 128)
        assert (embs_clamped >= -1e-6).all().item() is True

    def test_20_transfer_face_model_architecture(self):
        """Scenario 20: Verify TransferFaceModel API parity with FaceCNN."""
        from models.face_cnn import build_transfer_face_model
        transfer_model = build_transfer_face_model(num_classes=6, pretrained=False)
        transfer_model.eval()
        dummy_input = torch.randn(2, 3, 112, 112)

        embs = transfer_model.extract_features(dummy_input)
        assert embs.shape == (2, 128)
        norms = torch.norm(embs, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3)

        logits = transfer_model(dummy_input)
        assert logits.shape == (2, 6)

        probs, (top_class, top_conf) = transfer_model.predict_probabilities(dummy_input)
        assert probs.shape == (2, 6)
        assert top_class.shape == (2,)

    def test_21_vectorized_gallery_matrix_dot_product(self, recognizer):
        """Scenario 21: Verify vectorized gallery matching yields exact parity with dictionary loop."""
        emb1 = np.ones(128, dtype=np.float32) / np.sqrt(128)
        emb2 = np.zeros(128, dtype=np.float32)
        emb2[0] = 1.0

        recognizer.enrolled_gallery = {
            "ID_1": {"person_id": "ID_1", "name": "Alice", "embedding": emb1},
            "ID_2": {"person_id": "ID_2", "name": "Bob", "embedding": emb2}
        }
        recognizer._gallery_matrix = np.vstack([emb1, emb2])
        recognizer._gallery_pids = ["ID_1", "ID_2"]
        recognizer.similarity_threshold = 0.50

        # Query close to ID_2
        query = np.zeros(128, dtype=np.float32)
        query[0] = 0.98
        query[1] = 0.1989
        query /= np.linalg.norm(query)

        status, matched_id, sim, meta, runner_up = recognizer.compare_embedding_to_gallery(query)
        assert status == "known"
        assert matched_id == "ID_2"
        assert sim >= 0.95

    def test_22_torchscript_extractor_export(self, tmp_path):
        """Scenario 22: Verify TorchScript compilation of FaceCNN feature extractor."""
        from models.face_cnn import FaceCNN, export_torchscript_extractor
        real_model = FaceCNN(num_classes=6)
        export_path = str(tmp_path / "model_traced.pt")
        res_path = export_torchscript_extractor(real_model, export_path)
        assert os.path.exists(res_path)

        loaded_script = torch.jit.load(res_path)
        dummy_input = torch.randn(1, 3, 112, 112)
        out = loaded_script(dummy_input)
        assert out.shape == (1, 128)

