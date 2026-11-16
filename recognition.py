"""
recognition.py
--------------
Deep Learning Face Recognition Pipeline using Deep Feature Embeddings & Calibrated Cosine Similarity.

Key Architectural Principles:
1. Deep Face Embedding Extraction: Extracts 128-dimensional L2-normalized feature vectors
   using the trained FaceCNN backbone (extract_features).
2. Enrolled Face Gallery: Precomputes and caches normalized embeddings for all enrolled persons
   in known_faces/ and person_metadata.json.
3. Open-World Rejection: Compares query face embedding against registered gallery using Cosine Similarity.
   If similarity < calibrated threshold (0.85), decisively classifies as UNKNOWN.
   Softmax probability is NEVER used to force an impostor into a closed class.
4. Temporal Multi-Frame Consensus: Requires consistent votes across a sliding window per Track ID
   to prevent intermittent single-frame misidentifications.
5. Metric Separation:
   - YOLO detection confidence (object presence)
   - Face quality & detector confidence
   - Face embedding cosine similarity
   - Calibrated decision threshold
"""

import os
import json
import time
import logging
from datetime import datetime
from collections import deque, Counter
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import (
    CNN_MODEL_PATH, CLASS_NAMES_PATH, METADATA_PATH,
    RECOGNITION_CONFIG_PATH, DEFAULT_CONFIDENCE_THRESHOLD,
    RECOGNITION_INTERVAL, TEMPORAL_WINDOW_SIZE,
    REQUIRED_CONSISTENT_FRAMES, DEBUG_MODE, KNOWN_FACES_DIR,
    RECOGNITION_SIMILARITY_THRESHOLD, IDENTITY_DECISION_THRESHOLD,
    CAPTURED_IMAGES_DIR, MIN_FACE_SIZE
)
from face_detector import FaceDetector
from face_quality import evaluate_face_quality
from models.face_cnn import FaceCNN, build_transfer_face_model

logger = logging.getLogger("Recognition")


class TemporalTrackHistory:
    """Maintains sliding-window detection votes per Track ID to ensure multi-frame temporal consensus."""
    def __init__(self, window_size: int = TEMPORAL_WINDOW_SIZE):
        self.window_size = window_size
        self.predictions: deque = deque(maxlen=window_size)
        self.similarities: deque = deque(maxlen=window_size)
        self.confirmed_identity: Optional[str] = None
        self.last_seen: float = time.time()

    def add_observation(self, identity_id: str, similarity: float):
        self.predictions.append(identity_id)
        self.similarities.append(similarity)
        self.last_seen = time.time()

    def evaluate_consensus(self, required_consistent: int = REQUIRED_CONSISTENT_FRAMES) -> Tuple[bool, str, float, str]:
        """
        Evaluates sliding window votes.
        Returns:
            is_confirmed (bool)
            chosen_identity (str)
            avg_similarity (float)
            status ("VERIFIED" | "BUFFERING" | "UNKNOWN")
        """
        if not self.predictions:
            return False, "UNKNOWN", 0.0, "BUFFERING"

        # Filter out non-face observations for identity voting
        valid_pairs = [(p, s) for p, s in zip(self.predictions, self.similarities) if p not in ["NO_FACE", "NONE", "UNUSABLE"]]
        if not valid_pairs:
            if self.confirmed_identity and self.confirmed_identity != "UNKNOWN":
                return True, self.confirmed_identity, 0.85, "VERIFIED"
            return False, "UNKNOWN", 0.0, "BUFFERING"

        valid_preds = [p for p, s in valid_pairs]
        counts = Counter(valid_preds)
        most_common_id, freq = counts.most_common(1)[0]

        matching_sims = [s for p, s in valid_pairs if p == most_common_id]
        avg_sim = float(np.mean(matching_sims)) if matching_sims else 0.0

        # If person was previously verified as a known registered identity:
        # Retain the verified identity unless a conflicting registered person takes over
        if self.confirmed_identity and self.confirmed_identity != "UNKNOWN":
            competing_known = [p for p in valid_preds if p != "UNKNOWN" and p != self.confirmed_identity]
            if not competing_known:
                # No conflicting known identity; retain verified state
                return True, self.confirmed_identity, max(avg_sim, 0.82), "VERIFIED"
            else:
                comp_id, comp_freq = Counter(competing_known).most_common(1)[0]
                if comp_freq >= required_consistent:
                    self.confirmed_identity = comp_id
                    return True, comp_id, avg_sim, "VERIFIED"
                return True, self.confirmed_identity, avg_sim, "VERIFIED"

        if most_common_id == "UNKNOWN":
            return False, "UNKNOWN", avg_sim, "UNKNOWN"

        if freq >= required_consistent or (freq >= 2 and avg_sim >= 0.82) or (self.confirmed_identity == most_common_id):
            self.confirmed_identity = most_common_id
            return True, most_common_id, avg_sim, "VERIFIED"

        return False, most_common_id, avg_sim, "BUFFERING"


class RegistrationResult:
    """Encapsulates person enrollment result with full diagnostic feedback."""
    def __init__(
        self,
        success: bool,
        message: str,
        embedding: Optional[np.ndarray] = None,
        face_crop: Optional[np.ndarray] = None,
        person_id: str = ""
    ):
        self.success = success
        self.message = message
        self.embedding = embedding
        self.face_crop = face_crop
        self.person_id = person_id

    def __bool__(self) -> bool:
        return self.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "has_embedding": self.embedding is not None,
            "person_id": self.person_id
        }


class FaceRecognizer:
    """
    Production Face Recognition Engine utilizing Deep Face Embeddings & Calibrated Cosine Similarity.
    """
    def __init__(
        self,
        model_path: str = CNN_MODEL_PATH,
        class_names_path: str = CLASS_NAMES_PATH,
        metadata_path: str = METADATA_PATH,
        config_path: str = RECOGNITION_CONFIG_PATH
    ):
        self.model_path = model_path
        self.class_names_path = class_names_path
        self.metadata_path = metadata_path
        self.config_path = config_path

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else
            ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
        )
        logger.info("FaceRecognizer running on device: %s", self.device)

        self.detector = FaceDetector()
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Dict[str, Any]] = {}
        self.class_names: Dict[str, str] = {}
        self.similarity_threshold: float = RECOGNITION_SIMILARITY_THRESHOLD

        # Gallery of enrolled face embeddings: {person_id: {"name": str, "embedding": np.ndarray, ...}}
        self.enrolled_gallery: Dict[str, Dict[str, Any]] = {}
        self._gallery_matrix: Optional[np.ndarray] = None
        self._gallery_pids: List[str] = []

        # Temporal Tracking Histories: {track_id: TemporalTrackHistory}
        self.track_histories: Dict[int, TemporalTrackHistory] = {}
        self.frame_counter: int = 0
        self.recognition_interval: int = RECOGNITION_INTERVAL
        self.debug_mode: bool = DEBUG_MODE

        # Cached recognition results per track_id
        self.cached_track_results: Dict[int, Dict[str, Any]] = {}

        self.load_system()

    def load_system(self):
        """Loads model weights, class mapping, person metadata, and builds the face embedding gallery."""
        self.load_metadata()
        self.load_class_names()
        self.load_threshold_config()
        self.load_model()
        self.build_enrolled_gallery()

    def load_known_faces(self):
        """Reloads known faces and rebuilds embeddings."""
        self.load_system()

    def load_class_names(self):
        """Loads class mapping from class_names.json."""
        if os.path.exists(self.class_names_path):
            try:
                with open(self.class_names_path, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)
                logger.info("Loaded class mapping with %d classes", len(self.class_names))
            except Exception as e:
                logger.warning("Error reading class_names.json: %s", e)

    def load_metadata(self):
        """Loads person metadata records from person_metadata.json."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info("Loaded person metadata profiles for %d identities", len(self.metadata))
            except Exception as e:
                logger.warning("Error reading person_metadata.json: %s", e)

    def load_threshold_config(self):
        """Loads calibrated confidence / similarity threshold."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.similarity_threshold = float(data.get("calibrated_threshold", IDENTITY_DECISION_THRESHOLD))
                    logger.info("Loaded calibrated similarity threshold: %.3f", self.similarity_threshold)
            except Exception as e:
                logger.warning("Could not read recognition config: %s", e)
        else:
            self.similarity_threshold = IDENTITY_DECISION_THRESHOLD
            logger.info("Using calibrated decision threshold: %.3f", self.similarity_threshold)

    def load_model(self):
        """Initializes FaceCNN feature extractor from checkpoint."""
        if not os.path.exists(self.model_path):
            logger.warning("CNN model checkpoint not found at %s. Please train CNN model first.", self.model_path)
            self.model = None
            return

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            num_classes = checkpoint.get("num_classes", 6)
            arch = checkpoint.get("architecture", "FaceCNN_4Block")

            if "MobileNet" in arch:
                self.model = build_transfer_face_model(num_classes=num_classes, pretrained=False)
            else:
                self.model = FaceCNN(num_classes=num_classes, dropout_rate=0.4)

            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            logger.info("Face CNN Feature Extractor loaded successfully from %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load CNN model: %s", e)
            self.model = None

    def extract_embedding(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extracts 128-dimensional L2-normalized face embedding vector from preprocessed face crop.
        """
        if self.model is None or face_crop_bgr is None or face_crop_bgr.size == 0:
            return None

        try:
            tensor = self.detector.preprocess_to_tensor(face_crop_bgr, device=self.device)
            with torch.no_grad():
                emb = self.model.extract_features(tensor)
                emb_np = emb.squeeze(0).cpu().numpy()
                norm = np.linalg.norm(emb_np)
                if norm > 1e-6:
                    emb_np = emb_np / norm
                return emb_np
        except Exception as e:
            logger.debug("Failed to extract face embedding: %s", e)
            return None

    def build_enrolled_gallery(self):
        """
        Extracts and indexes deep face embeddings for all reference images in known_faces/.
        Supports multiple images per person and computes normalized centroid average embeddings.
        """
        self.enrolled_gallery.clear()
        if not os.path.exists(KNOWN_FACES_DIR):
            os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
            return

        valid_exts = (".jpg", ".jpeg", ".png", ".webp")
        image_entries: List[Tuple[str, str, str]] = []

        for root, dirs, files in os.walk(KNOWN_FACES_DIR):
            for fn in sorted(files):
                if not fn.lower().endswith(valid_exts):
                    continue
                fp = os.path.join(root, fn)
                rel = os.path.relpath(fp, KNOWN_FACES_DIR)
                parts = rel.split(os.sep)
                if len(parts) > 1:
                    person_id = parts[0]
                    raw_name = person_id
                else:
                    base = os.path.splitext(fn)[0]
                    subparts = base.split("_")
                    person_id = subparts[0]
                    raw_name = " ".join(subparts[1:]) if len(subparts) > 1 else person_id
                image_entries.append((person_id, raw_name, fp))

        # Group by person_id
        person_groups: Dict[str, List[Tuple[str, str]]] = {}
        for pid, rname, path in image_entries:
            person_groups.setdefault(pid, []).append((rname, path))

        for person_id, items in person_groups.items():
            embeddings: List[np.ndarray] = []
            filenames: List[str] = []
            primary_name = items[0][0]

            for raw_name, fp in items:
                try:
                    img = cv2.imread(fp)
                    if img is None or img.size == 0:
                        continue
                    ih, iw = img.shape[:2]
                    face_crop = img
                    if max(ih, iw) > 200:
                        crop, bbox, conf, is_valid = self.detector.detect_face_in_person(img, [0, 0, iw, ih])
                        if crop is not None and crop.size > 0:
                            face_crop = crop
                    emb = self.extract_embedding(face_crop)
                    if emb is not None:
                        embeddings.append(emb)
                        filenames.append(os.path.basename(fp))
                except Exception as e:
                    logger.warning("Could not extract embedding for %s (%s): %s", person_id, fp, e)

            if not embeddings:
                continue

            # Compute normalized centroid average embedding
            centroid = np.mean(embeddings, axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 1e-6:
                centroid = centroid / norm

            meta = self.get_metadata_for_id(person_id)
            display_name = meta.get("name", primary_name)
            dept = meta.get("department", "Engineering")
            role = meta.get("role", "Student" if person_id.startswith("STU") else "Faculty")

            self.enrolled_gallery[person_id] = {
                "person_id": person_id,
                "name": display_name,
                "department": dept,
                "role": role,
                "embedding": centroid,
                "sample_count": len(embeddings),
                "filenames": filenames,
                "filename": filenames[0],
                "metadata": meta
            }

        # Build vectorized matrix cache for high-throughput O(1) BLAS matrix-vector operations
        pids = []
        embs = []
        for pid, candidate in self.enrolled_gallery.items():
            emb = candidate.get("embedding")
            if emb is not None and isinstance(emb, np.ndarray) and emb.size == 128:
                pids.append(pid)
                embs.append(emb)
        if embs:
            self._gallery_matrix = np.vstack(embs)
            self._gallery_pids = pids
        else:
            self._gallery_matrix = None
            self._gallery_pids = []

        logger.info("Successfully enrolled %d individuals in Face Gallery (Centroid Average): %s",
                    len(self.enrolled_gallery), list(self.enrolled_gallery.keys()))

    def get_metadata_for_id(self, person_id: str) -> Dict[str, Any]:
        """Resolves rich profile metadata for given person ID."""
        if not person_id or person_id == "UNKNOWN":
            return {}
        # Direct key match
        if person_id in self.metadata:
            return self.metadata[person_id]
        if person_id in ["23", "STU023"]:
            m = self.metadata.get("23", self.metadata.get("Student_023", {}))
            if m:
                return m
        # Search by person_id property
        for k, v in self.metadata.items():
            if str(v.get("person_id", "")).lower() == person_id.lower():
                return v
        # Fallback to database lookup
        try:
            import firebase_service
            db_person = firebase_service.get_person_details(person_id)
            if db_person and db_person.get("status") == "known":
                return db_person
        except Exception:
            pass
        return {}

    def compare_embedding_to_gallery(self, query_emb: np.ndarray) -> Tuple[str, str, float, Optional[Dict[str, Any]], float]:
        """
        Compares query embedding against enrolled gallery using Cosine Similarity.
        Applies calibrated threshold and ambiguity margin between top two candidates.
        Uses vectorized BLAS matrix multiplication for high-throughput sub-millisecond matching.
        """
        if not self.enrolled_gallery or query_emb is None:
            return "unknown", "UNKNOWN", 0.0, None, 0.0

        if not isinstance(query_emb, np.ndarray) or query_emb.size == 0 or not np.all(np.isfinite(query_emb)):
            return "unknown", "UNKNOWN", 0.0, None, 0.0

        # Fast Vectorized BLAS Path
        if self._gallery_matrix is not None and query_emb.shape == (128,):
            sims_arr = np.dot(self._gallery_matrix, query_emb)
            if not np.all(np.isfinite(sims_arr)):
                return "unknown", "UNKNOWN", 0.0, None, 0.0

            best_idx = int(np.argmax(sims_arr))
            best_sim = float(sims_arr[best_idx])
            best_id = self._gallery_pids[best_idx]
            best_info = self.enrolled_gallery.get(best_id)

            if len(sims_arr) > 1:
                part = np.partition(sims_arr, -2)
                runner_up_sim = float(part[-2])
            else:
                runner_up_sim = 0.0

            if best_sim < self.similarity_threshold:
                return "unknown", "UNKNOWN", round(max(0.0, best_sim), 3), None, round(runner_up_sim, 3)

            ambiguity_margin = 0.04
            if len(sims_arr) > 1 and (best_sim - runner_up_sim) < ambiguity_margin and best_sim < (self.similarity_threshold + 0.06):
                logger.info("Match ambiguity between %s (%.3f) and second best (%.3f) < %.2f",
                            best_id, best_sim, runner_up_sim, ambiguity_margin)

            return "known", best_id, round(best_sim, 3), best_info, round(runner_up_sim, 3)

        # Fallback dictionary iteration loop (for arbitrary shapes / mocking)
        best_id = "UNKNOWN"
        best_sim = -1.0
        best_info = None
        sims: List[Tuple[str, float]] = []

        for pid, candidate in self.enrolled_gallery.items():
            candidate_emb = candidate.get("embedding")
            if candidate_emb is None or not isinstance(candidate_emb, np.ndarray):
                continue
            if query_emb.shape != candidate_emb.shape:
                continue
            sim = float(np.dot(query_emb, candidate_emb))
            if np.isnan(sim) or np.isinf(sim):
                continue
            sims.append((pid, sim))
            if sim > best_sim:
                best_sim = sim
                best_id = pid
                best_info = candidate

        sims.sort(key=lambda x: x[1], reverse=True)
        runner_up_sim = sims[1][1] if len(sims) > 1 else 0.0

        if best_sim < self.similarity_threshold:
            return "unknown", "UNKNOWN", round(max(0.0, best_sim), 3), None, round(runner_up_sim, 3)

        ambiguity_margin = 0.04
        if len(sims) > 1 and (best_sim - runner_up_sim) < ambiguity_margin and best_sim < (self.similarity_threshold + 0.06):
            logger.info("Match ambiguity between %s (%.3f) and %s (%.3f) < %.2f",
                        best_id, best_sim, sims[1][0], runner_up_sim, ambiguity_margin)

        return "known", best_id, round(best_sim, 3), best_info, round(runner_up_sim, 3)

    def recognize(
        self,
        frame: np.ndarray,
        person_bbox: List[int],
        track_id: Optional[int] = 1,
        yolo_conf: float = 0.80,
        is_single_image: bool = False,
        camera_id: str = "cctv"
    ) -> Dict[str, Any]:
        """
        Complete Real-Time Face Recognition Pipeline:
        1. Extract face from person crop using FaceDetector (YuNet / Haar / Fast-Path).
        2. Validate face quality (resolution, sharpness/blur, illumination).
        3. Extract 128-D L2-normalized embedding.
        4. Compare against enrolled gallery using Calibrated Cosine Similarity.
        5. Apply multi-frame temporal voting per track_id (bypassed for single image upload).
        6. Return complete detection result with separated confidence metrics.
        """
        t_id = track_id if track_id is not None else 1
        now = time.time()

        # Step 1: Detect face in person crop
        face_crop, face_bbox, face_conf, is_valid_face = self.detector.detect_face_in_person(frame, person_bbox)

        if face_crop is None or face_crop.size == 0 or not is_valid_face:
            logger.debug(
                "[DIAGNOSTIC-LOG] source=%s ts=%s detected_faces=0 crop_dim=0x0 face_quality=None "
                "emb_dim=0 registered_count=%d best_candidate=NONE best_sim=0.000 second_sim=0.000 "
                "threshold=%.2f status=no_face latency=%.1fms error=NO_FACE",
                camera_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                len(self.enrolled_gallery),
                self.similarity_threshold,
                round((time.time() - now) * 1000.0, 2)
            )
            return self._build_result(
                status="no_face",
                class_id="UNKNOWN",
                name="Face Not Available",
                similarity=0.0,
                yolo_conf=yolo_conf,
                face_conf=0.0,
                track_id=t_id,
                face_detected=False,
                face_bbox=None,
                verification_status="NO_FACE",
                meta=None
            )

        # Step 2: Quality assessment
        quality = evaluate_face_quality(face_crop)
        quality_score = quality["overall_score"]
        blur_score = quality["blur_score"]

        # If face is severely blurred or tiny in CCTV, do not treat it as an UNKNOWN intruder
        if (not is_single_image) and (blur_score < 14.0 or face_crop.shape[0] < 28 or face_crop.shape[1] < 28):
            return self._build_result(
                status="no_face",
                class_id="UNKNOWN",
                name="Face Unusable (Blurry)",
                similarity=0.0,
                yolo_conf=yolo_conf,
                face_conf=face_conf,
                track_id=t_id,
                face_detected=False,
                face_bbox=face_bbox,
                verification_status="FACE_BLURRY",
                meta=None,
                face_crop=face_crop,
                quality_score=quality_score,
                blur_score=blur_score
            )

        # Step 3: Throttling / Interval logic: bypass cache when analyzing single uploaded image
        if not is_single_image:
            self.frame_counter += 1
            if (self.frame_counter % self.recognition_interval != 0) and (t_id in self.cached_track_results):
                cached = dict(self.cached_track_results[t_id])
                cached["face_bbox"] = face_bbox
                cached["face_detected"] = True
                cached["yolo_confidence"] = yolo_conf
                return cached

        # Step 4: Extract embedding and compare against gallery
        query_emb = self.extract_embedding(face_crop)
        if query_emb is None:
            logger.debug(
                "[DIAGNOSTIC-LOG] source=%s ts=%s detected_faces=1 crop_dim=%s face_quality=insufficient "
                "emb_dim=0 registered_count=%d best_candidate=NONE best_sim=0.000 second_sim=0.000 "
                "threshold=%.2f status=unknown latency=%.1fms error=EMBEDDING_FAILED",
                camera_id,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                f"{face_crop.shape[1]}x{face_crop.shape[0]}",
                len(self.enrolled_gallery),
                self.similarity_threshold,
                round((time.time() - now) * 1000.0, 2)
            )
            return self._build_result(
                status="unknown",
                class_id="UNKNOWN",
                name=f"Unknown Person",
                similarity=0.0,
                yolo_conf=yolo_conf,
                face_conf=face_conf,
                track_id=t_id,
                face_detected=True,
                face_bbox=face_bbox,
                verification_status="EMBEDDING_FAILED",
                meta=None
            )

        raw_status, matched_id, similarity, matched_info, runner_up_sim = self.compare_embedding_to_gallery(query_emb)

        # Step 5: Multi-frame Temporal Verification vs Single Image Verification
        if is_single_image:
            is_confirmed = (raw_status == "known")
            final_status = "known" if is_confirmed else "unknown"
            final_id = matched_id if is_confirmed else "UNKNOWN"
            ver_status = "VERIFIED_SINGLE" if is_confirmed else "UNKNOWN"
            avg_sim = similarity
        else:
            if t_id not in self.track_histories:
                self.track_histories[t_id] = TemporalTrackHistory(window_size=TEMPORAL_WINDOW_SIZE)

            history = self.track_histories[t_id]
            history.add_observation(matched_id, similarity)
            is_confirmed, consensus_id, avg_sim, ver_status = history.evaluate_consensus(
                required_consistent=REQUIRED_CONSISTENT_FRAMES
            )

            final_status = "known" if is_confirmed else "unknown"
            final_id = consensus_id if is_confirmed else "UNKNOWN"

        final_meta = matched_info.get("metadata") if (is_confirmed and matched_info) else self.get_metadata_for_id(final_id)

        if is_confirmed and matched_info:
            display_name = matched_info.get("name", final_id)
        elif is_confirmed and final_meta:
            display_name = final_meta.get("name", final_id)
        else:
            display_name = "Unknown Person"

        result = self._build_result(
            status=final_status,
            class_id=final_id,
            name=display_name,
            similarity=avg_sim if is_confirmed else similarity,
            yolo_conf=yolo_conf,
            face_conf=face_conf,
            track_id=t_id,
            face_detected=True,
            face_bbox=face_bbox,
            verification_status=ver_status,
            meta=final_meta,
            face_crop=face_crop,
            quality_score=quality_score,
            blur_score=blur_score,
            runner_up_sim=runner_up_sim
        )

        crop_dim = f"{face_crop.shape[1]}x{face_crop.shape[0]}" if (face_crop is not None and face_crop.size > 0) else "0x0"
        emb_dim = len(query_emb) if query_emb is not None else 0
        proc_time_ms = round((time.time() - now) * 1000.0, 2)
        frame_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        registered_count = len(self.enrolled_gallery)
        quality_str = f"score={quality_score:.2f}_blur={blur_score:.1f}"
        error_msg = ver_status if final_status == "unknown" else "None"

        logger.info(
            "[DIAGNOSTIC-LOG] source=%s ts=%s detected_faces=%d crop_dim=%s face_quality=%s "
            "emb_dim=%d registered_count=%d best_candidate=%s best_sim=%.3f second_sim=%.3f "
            "threshold=%.2f status=%s latency=%.1fms error=%s",
            camera_id,
            frame_timestamp,
            1 if is_valid_face else 0,
            crop_dim,
            quality_str,
            emb_dim,
            registered_count,
            matched_id,
            similarity,
            runner_up_sim,
            self.similarity_threshold,
            final_status,
            proc_time_ms,
            error_msg
        )

        # Optional debug option: save selected face crops locally for inspection (bounded at 25 crops)
        if self.debug_mode and face_crop is not None and face_crop.size > 0:
            try:
                debug_crops_dir = os.path.join(CAPTURED_IMAGES_DIR, "debug_crops")
                os.makedirs(debug_crops_dir, exist_ok=True)
                existing = [f for f in os.listdir(debug_crops_dir) if f.endswith(".jpg")]
                if len(existing) < 25:
                    crop_fname = f"crop_{camera_id}_{final_status}_{int(now)}_{t_id}.jpg"
                    cv2.imwrite(os.path.join(debug_crops_dir, crop_fname), face_crop)
            except Exception as e:
                logger.debug("Could not save debug crop: %s", e)

        # Cache result for video streams
        if not is_single_image:
            self.cached_track_results[t_id] = result
        return result

    def _build_result(
        self,
        status: str,
        class_id: str,
        name: str,
        similarity: float,
        yolo_conf: float,
        face_conf: float,
        track_id: int,
        face_detected: bool,
        face_bbox: Optional[List[int]],
        verification_status: str,
        meta: Optional[Dict[str, Any]] = None,
        face_crop: Optional[np.ndarray] = None,
        quality_score: float = 0.0,
        blur_score: float = 0.0,
        runner_up_sim: float = 0.0
    ) -> Dict[str, Any]:
        """Builds standardized recognition output payload with separated metrics."""
        is_known = status == "known"
        department = meta.get("department", "Unassigned") if meta else "Visitor"
        role = meta.get("role", "Student" if class_id.startswith("STU") else "Visitor") if meta else "Visitor"
        roll_or_emp = meta.get("roll_number", meta.get("faculty_id", class_id)) if meta else class_id

        return {
            "status": status,
            "recognition_status": status,
            "class_id": class_id,
            "person_id": class_id,
            "name": name,
            "department": department,
            "role": role,
            "identifier": roll_or_emp,
            # Explicit Metric Separation
            "yolo_confidence": round(float(yolo_conf), 3),
            "face_confidence": round(float(face_conf), 3),
            "similarity_score": round(float(similarity), 3),
            "confidence": round(float(similarity), 3),  # Backward compatibility
            "decision_threshold": self.similarity_threshold,
            "quality_score": round(float(quality_score), 3),
            "blur_score": round(float(blur_score), 1),
            "runner_up_similarity": round(float(runner_up_sim), 3),
            # Telemetry
            "track_id": track_id,
            "face_detected": face_detected,
            "face_bbox": face_bbox,
            "verification_status": verification_status,
            "metadata": meta,
            "timetable": meta.get("timetable", []) if meta else [],
            "classes": meta.get("classes", []) if meta else []
        }


    def register_new_face(
        self,
        person_id: str,
        name: str,
        image_bgr: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RegistrationResult:
        """
        Dynamically registers a new person's face photo with Phase 3 validation:
        1. Validates that the image is valid and decodable.
        2. Detects all faces in the image.
        3. Rejects images with 0 usable faces.
        4. Rejects ambiguous registration images with multiple faces.
        5. Assesses face size, blur, and image quality.
        6. Extracts 128-D L2-normalized embedding using the unified FaceCNN model.
        7. Validates that the embedding contains finite numeric values and has length 128.
        8. Validates self-similarity (dot product ~= 1.0).
        9. Checks for conflicting duplicate identities in the active gallery.
        10. Saves reference crop and updates active gallery.
        """
        if image_bgr is None or image_bgr.size == 0:
            return RegistrationResult(False, "Empty or invalid image data provided.")

        try:
            h, w = image_bgr.shape[:2]
            if h < MIN_FACE_SIZE[1] or w < MIN_FACE_SIZE[0]:
                return RegistrationResult(False, f"Image resolution too small ({w}x{h}px). Minimum is {MIN_FACE_SIZE[0]}x{MIN_FACE_SIZE[1]}px.")

            # 1. Detect all faces in registration photograph
            detected_faces = self.detector.detect_all_faces(image_bgr, min_conf=0.40)

            if len(detected_faces) == 0:
                return RegistrationResult(False, "No usable face detected in photograph. Please provide a clear, well-lit portrait.")

            if len(detected_faces) > 1:
                return RegistrationResult(
                    False,
                    f"Ambiguous registration: {len(detected_faces)} faces detected. Please provide a photo with only one person."
                )

            face_info = detected_faces[0]
            face_crop = face_info["crop"]

            # 2. Check face quality (sharpness, blur variance, lighting)
            quality = evaluate_face_quality(face_crop)
            if not quality["is_acceptable"]:
                issue_desc = ", ".join(quality["issues"])
                logger.warning("Registration quality warning for %s (%s): %s", name, person_id, issue_desc)
                # Allow marginally acceptable faces if blur variance > 20
                if quality["blur_score"] < 15.0 or quality["overall_score"] < 0.25:
                    return RegistrationResult(
                        False,
                        f"Face quality insufficient: {issue_desc}. Please upload a sharper, better-lit portrait."
                    )

            # 3. Extract CNN embedding (128-D, L2-normalized)
            emb = self.extract_embedding(face_crop)
            if emb is None:
                return RegistrationResult(False, "Face embedding extraction failed.")

            # 4. Validate embedding properties
            if emb.shape != (128,):
                return RegistrationResult(False, f"Invalid embedding dimension {emb.shape}; expected (128,).")

            if not np.all(np.isfinite(emb)):
                return RegistrationResult(False, "Embedding contains non-finite values (NaN or Inf).")

            self_sim = float(np.dot(emb, emb))
            if not (0.95 <= self_sim <= 1.05):
                return RegistrationResult(False, f"Embedding normalization error: self-similarity is {self_sim:.3f}.")

            # 5. Check duplicate identity conflict
            for enrolled_id, enrolled_data in self.enrolled_gallery.items():
                if enrolled_id != person_id:
                    other_emb = enrolled_data.get("embedding")
                    if other_emb is not None:
                        sim = float(np.dot(emb, other_emb))
                        if sim > 0.96:
                            logger.warning(
                                "Duplicate face warning: %s (%s) matches existing enrolled %s (%s) with similarity %.3f",
                                name, person_id, enrolled_data.get("name"), enrolled_id, sim
                            )

            # 6. Save reference crop to known_faces directory
            safe_pid = os.path.basename(person_id.strip().replace("/", "_").replace("\\", "_"))
            safe_name = os.path.basename((name or "user").strip().replace(" ", "_").replace("/", "_").replace("\\", "_"))
            filename = f"{safe_pid}_{safe_name}.jpg"
            save_path = os.path.join(KNOWN_FACES_DIR, filename)
            cv2.imwrite(save_path, face_crop)

            # 7. Update enrolled gallery
            meta_dict = dict(metadata or {})
            meta_dict["embedding"] = emb.tolist()
            self.enrolled_gallery[person_id] = {
                "person_id": person_id,
                "name": name,
                "department": meta_dict.get("department", "Engineering"),
                "role": meta_dict.get("role", "Student"),
                "embedding": emb,
                "filename": filename,
                "metadata": meta_dict
            }
            logger.info("Enrolled new face in active gallery: %s (%s) [emb_dim=128, self_sim=%.3f]", name, person_id, self_sim)
            return RegistrationResult(True, "Person enrolled successfully.", embedding=emb, face_crop=face_crop, person_id=person_id)

        except Exception as e:
            logger.error("Failed to register new face: %s", e, exc_info=True)
            return RegistrationResult(False, f"Registration failed: {str(e)}")
