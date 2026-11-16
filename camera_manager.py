"""
camera_manager.py
-----------------
Central Coordinator and Pipeline Orchestrator for Real-Time AI Camera Surveillance.

Coordinates:
1. RTSP & CCTV Stream Management via rtsp_camera.RTSPCamera
2. Multi-camera scalability with dynamic camera addition and switching
3. YOLO Object Detection (persons, mobile phones, laptops, backpacks, vehicles, etc.)
4. Deep Face Recognition (embeddings, calibrated cosine similarity, open-world rejection)
5. Face Quality Gating (blur, size, lighting, aspect ratio)
6. Event-based detection logging with cooldown de-duplication to prevent DB flooding
7. MJPEG live streaming with visual telemetry and bounding box overlays
8. Firebase Firestore and local image synchronization
"""

import os
import cv2
import time
import uuid
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List, Set, Tuple
import numpy as np

from rtsp_camera import RTSPCamera, mask_rtsp_url
from yolo_detector import YOLODetector
from recognition import FaceRecognizer
from face_quality import evaluate_face_quality
import firebase_service
from config import (
    CAPTURED_IMAGES_DIR, DETECTION_COOLDOWN_SECONDS,
    MAX_MISSING_FRAMES, PROCESS_EVERY_N_FRAMES,
    YOLO_CONFIDENCE_THRESHOLD, RECOGNITION_SIMILARITY_THRESHOLD,
    IDENTITY_DECISION_THRESHOLD, ALERT_ON_UNKNOWN_PERSON,
    SPATIAL_DEDUPLICATION_IOU
)

logger = logging.getLogger("CameraManager")

# Colors for bounding box annotations (BGR format)
COLOR_KNOWN = (34, 197, 94)       # Emerald Green
COLOR_UNKNOWN = (239, 68, 68)     # Crimson Red
COLOR_ELECTRONIC = (249, 115, 22) # Amber / Orange (Mobile phone, Laptop)
COLOR_VEHICLE = (59, 130, 246)    # Royal Blue (Car, Motorcycle, Bus)
COLOR_OBJECT = (168, 85, 247)     # Purple (Backpack, Bottle, Chair)
COLOR_NO_FACE = (148, 163, 184)   # Slate / Muted Gray (Face unavailable)
COLOR_HUD_BG = (15, 23, 42)       # Dark Slate


class TrackedEntity:
    """Represents an active presence session for a detected object or person."""
    def __init__(self, session_key: str, entity_type: str, person_id: Optional[str],
                 name: str, class_name: str, track_id: Optional[int]):
        self.session_key = session_key
        self.entity_type = entity_type  # "KNOWN", "UNKNOWN", "ELECTRONIC", "VEHICLE", "OBJECT"
        self.person_id = person_id
        self.name = name
        self.class_name = class_name
        self.track_id = track_id
        self.captured = False
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.missing_frames = 0
        self.capture_data: Optional[Dict[str, Any]] = None


class CameraManager:
    """
    Singleton Manager orchestrating multiple RTSP/CCTV cameras and the complete AI pipeline.
    """
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        with cls._singleton_lock:
            if cls._instance is None:
                cls._instance = super(CameraManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Multi-camera registry: {camera_id: RTSPCamera}
        self.cameras: Dict[str, RTSPCamera] = {}
        self.active_camera_id: str = os.environ.get("CAMERA_ID", "gate_camera_1").strip()

        # Cooldown & tracking
        self.cooldown_seconds = DETECTION_COOLDOWN_SECONDS
        self.process_every_n_frames = PROCESS_EVERY_N_FRAMES
        self.last_logged_time: Dict[str, float] = {}  # {session_key: timestamp}
        self.active_sessions: Dict[str, TrackedEntity] = {}
        self.unknown_spatial_sessions: Dict[str, List[Dict[str, Any]]] = {}
        self.session_lock = threading.Lock()

        # Frames & streaming
        self.current_annotated_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.is_processing = False
        self._processor_thread: Optional[threading.Thread] = None

        # Telemetry
        self.processed_fps = 0.0
        self._proc_frame_count = 0
        self._last_fps_time = time.time()
        self.frame_counter = 0

        # Latest event telemetry for UI popups
        self.latest_recognition: Optional[Dict[str, Any]] = None
        self.recent_alerts: List[Dict[str, Any]] = []

        # AI Pipelines
        self.yolo = YOLODetector(confidence_threshold=YOLO_CONFIDENCE_THRESHOLD)
        self.recognizer = FaceRecognizer()

        # Initialize default cameras from environment
        self._init_default_cameras()

        # Start background processor thread
        self.start()

    def _init_default_cameras(self):
        """Pre-configures default camera streams from environment or fallback."""
        default_rtsp = os.environ.get("RTSP_URL", "").strip()
        cam1_url = os.environ.get("RTSP_URL_GATE_1", default_rtsp)
        cam1_loc = os.environ.get("RTSP_LOCATION_GATE_1", os.environ.get("CAMERA_LOCATION", "Main Gate Entrance"))
        cam1_name = os.environ.get("RTSP_NAME_GATE_1", os.environ.get("CAMERA_NAME", "Main Gate CCTV"))

        cam2_url = os.environ.get("RTSP_URL_GATE_2", "")
        cam2_loc = os.environ.get("RTSP_LOCATION_GATE_2", "North Gate Entrance")
        cam2_name = os.environ.get("RTSP_NAME_GATE_2", "North Gate CCTV")

        self.add_camera("gate_camera_1", cam1_url, cam1_name, cam1_loc)
        self.add_camera("gate_camera_2", cam2_url, cam2_name, cam2_loc)

    def add_camera(self, camera_id: str, source: str, name: str = "CCTV Camera", location: str = "Campus") -> RTSPCamera:
        """Registers a new or existing camera."""
        if camera_id in self.cameras:
            cam = self.cameras[camera_id]
            cam.update_source(source, name, location)
            return cam

        cam = RTSPCamera(camera_id=camera_id, source=source, name=name, location=location)
        self.cameras[camera_id] = cam
        logger.info("Registered camera [%s]: %s -> %s", camera_id, name, mask_rtsp_url(source))
        return cam

    def get_active_camera(self) -> Optional[RTSPCamera]:
        """Returns currently active RTSPCamera instance."""
        return self.cameras.get(self.active_camera_id)

    def switch_camera(self, camera_id: str) -> bool:
        """Switches the active camera feed to another registered camera ID."""
        if camera_id not in self.cameras:
            logger.warning("Attempted to switch to unknown camera_id: %s", camera_id)
            return False

        old_id = self.active_camera_id
        self.active_camera_id = camera_id

        # Ensure target camera is running
        target_cam = self.cameras[camera_id]
        if not target_cam.is_running:
            target_cam.start()

        logger.info("Switched active camera from [%s] to [%s]", old_id, camera_id)
        return True

    def start(self):
        """Starts active camera ingestion and AI frame processing thread."""
        active_cam = self.get_active_camera()
        if active_cam and not active_cam.is_running:
            active_cam.start()

        if not self.is_processing:
            self.is_processing = True
            self._processor_thread = threading.Thread(
                target=self._ai_processor_loop,
                daemon=True,
                name="CameraManagerProcessor"
            )
            self._processor_thread.start()
            logger.info("CameraManager background processing thread started.")

    def stop(self):
        """Stops active camera and pauses processing."""
        active_cam = self.get_active_camera()
        if active_cam:
            active_cam.stop()

    def disconnect_active(self):
        """Disconnects active camera."""
        active_cam = self.get_active_camera()
        if active_cam:
            active_cam.disconnect()

    def _ai_processor_loop(self):
        """
        Continuous background processing loop.
        Drains raw frames from active RTSPCamera, performs inference (throttled),
        draws rich bounding box overlays, handles presence cooldowns, and updates display buffer.
        """
        while self.is_processing:
            active_cam = self.get_active_camera()
            if active_cam is None or not active_cam.is_running:
                # Idle briefly if camera offline
                time.sleep(0.05)
                continue

            ret, frame = active_cam.read_frame()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.02)
                continue

            self.frame_counter += 1
            h, w = frame.shape[:2]

            # Throttle AI compute to every N frames to conserve CPU/GPU
            run_inference = (self.frame_counter % self.process_every_n_frames == 0)

            annotated_frame = frame.copy()
            active_keys_this_frame: Set[str] = set()

            if run_inference:
                try:
                    # 1. Run YOLO Object Detection (persons, mobile phones, laptops, etc.)
                    detections = self.yolo.detect(frame)

                    for det in detections:
                        cls_name = det["class_name"]
                        conf = det["confidence"]
                        bbox = det["bbox"]
                        track_id = det["track_id"] or 1
                        x1, y1, x2, y2 = bbox

                        # ---------------------------------------------------------
                        # Case A: Detected Person -> Run Face Recognition
                        # ---------------------------------------------------------
                        if det["is_person"]:
                            rec_result = self.recognizer.recognize(
                                frame=frame,
                                person_bbox=bbox,
                                track_id=track_id,
                                yolo_conf=conf,
                                camera_id=active_cam.camera_id
                            )

                            status = rec_result["status"]  # "known" or "unknown"
                            person_id = rec_result["person_id"]
                            person_name = rec_result["name"]
                            sim_score = rec_result["similarity_score"]
                            ver_status = rec_result["verification_status"]
                            face_bbox = rec_result["face_bbox"]

                            if status == "known":
                                session_key = f"person_{person_id}"
                                box_color = COLOR_KNOWN
                                label_text = f"{person_name} ({sim_score:.0%})"
                                badge_text = f"KNOWN | ID: {person_id}"
                            elif status == "unknown":
                                session_key = self._resolve_unknown_session(active_cam.camera_id, bbox, time.time())
                                box_color = COLOR_UNKNOWN
                                label_text = f"UNKNOWN ({sim_score:.0%})"
                                badge_text = "UNKNOWN VISITOR"
                            else:
                                session_key = f"noface_{active_cam.camera_id}_{track_id}"
                                box_color = COLOR_NO_FACE
                                label_text = f"PERSON ({conf:.0%})"
                                badge_text = "FACE NOT AVAILABLE"

                            active_keys_this_frame.add(session_key)

                            # Draw Person Bounding Box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                            # Draw Face Bounding Box if detected
                            if face_bbox:
                                fx1, fy1, fx2, fy2 = face_bbox
                                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)

                            # Label Banner
                            self._draw_label_banner(annotated_frame, x1, y1, label_text, badge_text, box_color)

                            # Cooldown & Event Logging
                            self._evaluate_logging(
                                session_key=session_key,
                                entity_type="KNOWN" if status == "known" else ("UNKNOWN" if status == "unknown" else "FACE_UNAVAILABLE"),
                                person_id=person_id if status == "known" else ("UNKNOWN" if status == "unknown" else None),
                                person_name=person_name if status == "known" else ("Unknown Visitor" if status == "unknown" else "Person (Face Unavailable)"),
                                class_name="person",
                                track_id=track_id,
                                yolo_conf=conf,
                                face_conf=rec_result.get("face_confidence", 0.0),
                                similarity_score=sim_score,
                                decision_threshold=rec_result.get("decision_threshold", IDENTITY_DECISION_THRESHOLD),
                                status=status,
                                frame_bgr=annotated_frame,
                                crop_bbox=bbox,
                                face_bbox=face_bbox,
                                active_cam=active_cam,
                                meta=rec_result.get("metadata"),
                                raw_frame_bgr=frame
                            )

                            self.latest_recognition = rec_result

                        # ---------------------------------------------------------
                        # Case B: Detected Electronic / Object (Mobile Phone, Laptop, etc.)
                        # ---------------------------------------------------------
                        else:
                            if det.get("is_electronic", False):
                                box_color = COLOR_ELECTRONIC
                                category_label = "ELECTRONIC"
                            elif det.get("is_vehicle", False):
                                box_color = COLOR_VEHICLE
                                category_label = "VEHICLE"
                            else:
                                box_color = COLOR_OBJECT
                                category_label = "OBJECT"

                            session_key = f"obj_{cls_name}_{track_id}"
                            active_keys_this_frame.add(session_key)

                            label_text = f"{cls_name.upper()} ({conf:.0%})"
                            badge_text = category_label

                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
                            self._draw_label_banner(annotated_frame, x1, y1, label_text, badge_text, box_color)

                            # Log object detection with cooldown
                            self._evaluate_logging(
                                session_key=session_key,
                                entity_type=category_label,
                                person_id=None,
                                person_name=cls_name.capitalize(),
                                class_name=cls_name,
                                track_id=track_id,
                                yolo_conf=conf,
                                face_conf=0.0,
                                similarity_score=0.0,
                                decision_threshold=self.yolo.confidence_threshold,
                                status="object",
                                frame_bgr=annotated_frame,
                                crop_bbox=bbox,
                                face_bbox=None,
                                active_cam=active_cam,
                                raw_frame_bgr=frame
                            )

                    # Manage missing sessions
                    self._sweep_missing_sessions(active_keys_this_frame)

                except Exception as proc_err:
                    logger.error("Error during AI frame processing: %s", proc_err)

            # Draw HUD overlay (FPS, Camera Name, Status)
            self._draw_hud(annotated_frame, active_cam)

            # Store finished annotated frame in thread-safe buffer
            with self.frame_lock:
                self.current_annotated_frame = annotated_frame

            # Performance telemetry
            self._proc_frame_count += 1
            now = time.time()
            if now - self._last_fps_time >= 1.0:
                self.processed_fps = round(self._proc_frame_count / (now - self._last_fps_time), 1)
                self._proc_frame_count = 0
                self._last_fps_time = now

            # Modest sleep to prevent 100% CPU lock
            time.sleep(0.005)

    def _draw_label_banner(self, img: np.ndarray, x: int, y: int, title: str, subtitle: str, color: Tuple[int, int, int]):
        """Renders anti-aliased header badge above bounding box."""
        h, w = img.shape[:2]
        banner_w = max(130, len(title) * 9 + 20)
        banner_h = 36

        by1 = max(0, y - banner_h)
        by2 = y
        bx1 = max(0, x)
        bx2 = min(w, bx1 + banner_w)

        # Semi-transparent background with zero-copy localized ROI blending
        if by2 > by1 and bx2 > bx1:
            roi = img[by1:by2, bx1:bx2]
            overlay_roi = np.full_like(roi, color)
            cv2.addWeighted(overlay_roi, 0.85, roi, 0.15, 0, roi)

        # Text labels
        cv2.putText(img, title, (bx1 + 6, by1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, subtitle, (bx1 + 6, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

    def _draw_hud(self, img: np.ndarray, cam: RTSPCamera):
        """Draws top status bar with FPS, Camera ID, and AI status."""
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 32), COLOR_HUD_BG, -1)

        status_text = f"CAM: {cam.name} [{cam.camera_id}] | STATUS: {cam.connection_status} | FPS: {self.processed_fps:.1f} | AI: YOLOv8 + CNN Embeddings"
        cv2.putText(img, status_text, (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)

        # Timestamp on right
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, ts_str, (max(10, w - 170), 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

    def _resolve_unknown_session(self, cam_id: str, bbox: List[int], now: float) -> str:
        """
        Spatially deduplicates unknown persons across consecutive frames and tracker drops.
        Prevents synthetic UNKNOWN-XXXX IDs and eliminates duplicate alert floods.
        """
        sessions = self.unknown_spatial_sessions.setdefault(cam_id, [])
        # Prune dead sessions (> 8.0s unseen)
        sessions[:] = [s for s in sessions if (now - s["last_seen"]) < 8.0]
        for s in sessions:
            if self._compute_bbox_iou(s["bbox"], bbox) >= SPATIAL_DEDUPLICATION_IOU:
                s["last_seen"] = now
                s["bbox"] = bbox
                return s["session_id"]

        # New spatial session for a genuinely new visitor
        new_id = f"unknown_{cam_id}_{uuid.uuid4().hex[:6]}"
        sessions.append({
            "session_id": new_id,
            "bbox": bbox,
            "last_seen": now
        })
        return new_id

    @staticmethod
    def _compute_bbox_iou(boxA: List[int], boxB: List[int]) -> float:
        """Calculates Intersection-over-Union between two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
        return interArea / float(boxAArea + boxBArea - interArea)

    def _evaluate_logging(
        self,
        session_key: str,
        entity_type: str,
        person_id: Optional[str],
        person_name: str,
        class_name: str,
        track_id: int,
        yolo_conf: float,
        face_conf: float,
        similarity_score: float,
        decision_threshold: float,
        status: str,
        frame_bgr: np.ndarray,
        crop_bbox: List[int],
        face_bbox: Optional[List[int]] = None,
        active_cam: Optional[RTSPCamera] = None,
        meta: Optional[Dict[str, Any]] = None,
        raw_frame_bgr: Optional[np.ndarray] = None
    ):
        """
        Atomic evaluation of cooldown and event logging:
        If within cooldown window, skips logging to avoid flooding database.
        If cooldown expired or new session, creates snapshot and commits detection event.
        """
        if active_cam is None:
            active_cam = self.get_active_camera()
            if active_cam is None:
                return

        now = time.time()
        last_time = self.last_logged_time.get(session_key, 0.0)

        # Check cooldown
        if (now - last_time) < self.cooldown_seconds:
            return

        self.last_logged_time[session_key] = now
        event_id = str(uuid.uuid4())

        # Determine source frame for clean crop thumbnail (raw unannotated frame if available)
        source_frame = raw_frame_bgr if raw_frame_bgr is not None else frame_bgr
        ch, cw = source_frame.shape[:2]

        # Determine thumbnail crop region:
        # If person and face_bbox available, crop face.
        # If person without face_bbox or object, crop object/person bbox (NEVER a face crop for a phone).
        if status in ["known", "unknown"] and face_bbox:
            cx1, cy1, cx2, cy2 = face_bbox
            pad_x = int((cx2 - cx1) * 0.15)
            pad_y = int((cy2 - cy1) * 0.15)
        else:
            cx1, cy1, cx2, cy2 = crop_bbox
            pad_x = int((cx2 - cx1) * 0.08)
            pad_y = int((cy2 - cy1) * 0.08)

        tcx1 = max(0, cx1 - pad_x)
        tcy1 = max(0, cy1 - pad_y)
        tcx2 = min(cw, cx2 + pad_x)
        tcy2 = min(ch, cy2 + pad_y)
        thumb_bgr = source_frame[tcy1:tcy2, tcx1:tcx2]
        if thumb_bgr.size == 0:
            thumb_bgr = source_frame

        norm_status = "face_unavailable" if status in ["no_face", "face_unavailable"] else status

        # Save both full evidence frame and clean crop thumbnail tied to event_id
        ev_id, ev_path, ev_url, th_id, th_path, th_url = self._save_event_evidence(
            event_id=event_id,
            frame_bgr=frame_bgr,
            thumb_bgr=thumb_bgr,
            camera_id=active_cam.camera_id,
            identifier=(person_id if (status == "known" and person_id) else (class_name if status == "object" else norm_status))
        )

        # Configurable alert categorization
        if status == "known":
            event_type = "known_person_entry"
        elif status == "unknown":
            event_type = "unknown_security_alert" if ALERT_ON_UNKNOWN_PERSON else "unknown_visitor"
        elif norm_status == "face_unavailable":
            event_type = "person_face_unavailable"
        else:
            event_type = f"object_detected_{class_name}"

        record = {
            "id": event_id,
            "camera_id": active_cam.camera_id,
            "camera_name": active_cam.name,
            "camera_location": active_cam.location,
            "object_class": class_name,
            "object_type": class_name,
            "recognition_status": norm_status,
            "person_id": person_id if status == "known" else ("UNKNOWN" if status == "unknown" else None),
            "person_name": person_name,
            "department": meta.get("department") if meta else ("General" if status == "known" else ("-" if status != "unknown" else "Visitor")),
            "role": meta.get("role") if meta else ("General" if status == "known" else ("-" if status != "unknown" else "Visitor")),
            # Explicit confidence metrics
            "yolo_confidence": round(float(yolo_conf), 3),
            "face_confidence": round(float(face_conf), 3),
            "similarity_score": round(float(similarity_score), 3),
            "confidence": round(float(similarity_score if status in ['known', 'unknown'] else yolo_conf), 3),
            "decision_threshold": round(float(decision_threshold), 3),
            # Evidence & Thumbnails
            "image_id": ev_id,
            "image_url": ev_url,
            "thumbnail_url": th_url,
            "local_image_path": ev_path,
            "local_thumb_path": th_path,
            "event_type": event_type,
            "timestamp": datetime.now().strftime("%d %b %Y, %I:%M:%S %p"),
            "raw_timestamp": datetime.now().isoformat()
        }

        # Save to Firebase and local store
        firebase_service.save_detection(record)
        self.recent_alerts.insert(0, record)
        if len(self.recent_alerts) > 50:
            self.recent_alerts.pop()

        logger.info("[EVENT LOGGED] %s | Type: %s | Camera: %s | Score: %.2f",
                    person_name, event_type, active_cam.camera_id, similarity_score or yolo_conf)

    def _save_event_evidence(
        self,
        event_id: str,
        frame_bgr: np.ndarray,
        thumb_bgr: np.ndarray,
        camera_id: str,
        identifier: str
    ) -> Tuple[str, str, str, str, str, str]:
        """
        Saves JPEG full evidence frame and clean crop thumbnail locally and syncs to Firestore.
        Tied to unique event_id to prevent overwrite collisions.
        """
        clean_id = identifier.replace(" ", "_").replace("/", "-")
        ev_image_id = f"evidence_{camera_id}_{clean_id}_{event_id[:8]}"
        th_image_id = f"thumb_{camera_id}_{clean_id}_{event_id[:8]}"

        ev_filename = f"{ev_image_id}.jpg"
        th_filename = f"{th_image_id}.jpg"

        ev_local_path = os.path.join(CAPTURED_IMAGES_DIR, ev_filename)
        th_local_path = os.path.join(CAPTURED_IMAGES_DIR, th_filename)

        try:
            cv2.imwrite(ev_local_path, frame_bgr)
            cv2.imwrite(th_local_path, thumb_bgr)

            ev_res = firebase_service.save_image_to_firestore(
                image_input=frame_bgr,
                image_id=ev_image_id,
                filename=ev_filename,
                metadata={"camera_id": camera_id, "type": "evidence", "event_id": event_id}
            )
            th_res = firebase_service.save_image_to_firestore(
                image_input=thumb_bgr,
                image_id=th_image_id,
                filename=th_filename,
                metadata={"camera_id": camera_id, "type": "thumbnail", "event_id": event_id}
            )

            ev_url = ev_res.get("image_url", f"/captured_images/{ev_filename}")
            th_url = th_res.get("image_url", f"/captured_images/{th_filename}")
            return ev_image_id, ev_local_path, ev_url, th_image_id, th_local_path, th_url
        except Exception as e:
            logger.error("Failed saving evidence snapshots: %s", e)
            return (
                ev_image_id, ev_local_path, f"/captured_images/{ev_filename}",
                th_image_id, th_local_path, f"/captured_images/{th_filename}"
            )

    def _save_evidence_image(self, frame_bgr: np.ndarray, camera_id: str, identifier: str) -> Tuple[str, str, str]:
        """Legacy helper maintained for backward compatibility."""
        clean_id = identifier.replace(" ", "_").replace("/", "-")
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_id = f"{camera_id}_{clean_id}_{time_str}"
        filename = f"{image_id}.jpg"
        local_path = os.path.join(CAPTURED_IMAGES_DIR, filename)

        try:
            cv2.imwrite(local_path, frame_bgr)
            save_res = firebase_service.save_image_to_firestore(
                image_input=frame_bgr,
                image_id=image_id,
                filename=filename,
                metadata={"camera_id": camera_id, "identifier": clean_id}
            )
            firestore_url = save_res.get("image_url", f"/captured_images/{filename}")
            return image_id, local_path, firestore_url
        except Exception as e:
            logger.error("Failed saving evidence snapshot: %s", e)
            return image_id, local_path, f"/captured_images/{filename}"

    def _sweep_missing_sessions(self, active_keys: Set[str]):
        """Removes sessions not seen for multiple frames."""
        with self.session_lock:
            to_remove = []
            for key, entity in self.active_sessions.items():
                if key not in active_keys:
                    entity.missing_frames += 1
                    if entity.missing_frames >= MAX_MISSING_FRAMES:
                        to_remove.append(key)
            for k in to_remove:
                self.active_sessions.pop(k, None)

    def get_latest_frame_jpeg(self) -> Optional[bytes]:
        """Encodes the latest annotated frame as JPEG for MJPEG streaming."""
        with self.frame_lock:
            frame = self.current_annotated_frame
            if frame is None or frame.size == 0:
                # If no frame yet, pull directly from active camera
                active_cam = self.get_active_camera()
                if active_cam:
                    ret, raw = active_cam.read_frame()
                    if ret and raw is not None:
                        frame = raw

        if frame is not None and frame.size > 0:
            ret, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return buf.tobytes()
        return None

    def get_status(self) -> Dict[str, Any]:
        """Provides full multi-camera and AI telemetry dictionary."""
        active_cam = self.get_active_camera()
        active_cam_status = active_cam.get_status() if active_cam else {}

        # Count active/connected cameras
        connected_cams = sum(1 for c in self.cameras.values() if c.connection_status in ["CONNECTED", "CONNECTED (WEBCAM)"])
        total_cams = len(self.cameras)

        return {
            "active_camera_id": self.active_camera_id,
            "connected_cameras_count": connected_cams,
            "total_cameras_count": total_cams,
            "cameras_summary": f"{connected_cams:02d} / {total_cams:02d} Online",
            "camera": active_cam_status,
            "fps": self.processed_fps,
            "is_processing": self.is_processing,
            "yolo_status": self.yolo.get_status(),
            "enrolled_persons_count": len(self.recognizer.enrolled_gallery),
            "enrolled_persons": list(self.recognizer.enrolled_gallery.keys()),
            "cooldown_seconds": self.cooldown_seconds
        }


def generate_mjpeg_stream(camera_manager: Optional[CameraManager] = None):
    """Generator yielding multipart MJPEG frames for Flask video_feed endpoint."""
    cm = camera_manager or CameraManager()
    while True:
        jpg_bytes = cm.get_latest_frame_jpeg()
        if jpg_bytes is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )
        time.sleep(0.033)  # Approx 30 FPS cap
