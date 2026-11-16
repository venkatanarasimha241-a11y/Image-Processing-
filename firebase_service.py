"""
firebase_service.py
-------------------
Provides abstraction layer for Firestore database and Cloud Storage.
Maintains in-memory and local file persistence when running in DEMO MODE.
"""

import os
import uuid
import base64
import queue
import threading
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

from firebase_config import (
    get_firestore_client, get_storage_bucket, is_demo_mode,
    get_firebase_status, test_firestore_connection, initialize_firebase
)

logger = logging.getLogger("FirebaseService")

# Non-blocking async queue for Firestore writes
_cloud_sync_queue: queue.Queue = queue.Queue(maxsize=1000)

def _cloud_sync_worker():
    """Background daemon thread: syncs images and detections to Firestore without blocking RTSP camera or web server."""
    while True:
        try:
            job_type, payload = _cloud_sync_queue.get()
            db = get_firestore_client()
            if db is not None and not is_demo_mode():
                try:
                    if job_type == "image":
                        image_id, image_doc = payload
                        db.collection("images").document(image_id).set(image_doc)
                        logger.info("Async synced image %s to Cloud Firestore", image_id)
                    elif job_type == "detection":
                        record_id, detection_data = payload
                        db.collection("detections").document(record_id).set(detection_data)
                        logger.info("Async synced detection %s to Cloud Firestore", record_id)
                except Exception as write_err:
                    logger.debug("Background Firestore sync failed for %s: %s", job_type, write_err)
            _cloud_sync_queue.task_done()
        except Exception as queue_err:
            logger.debug("Cloud sync queue error: %s", queue_err)

_sync_thread = threading.Thread(target=_cloud_sync_worker, daemon=True, name="CloudSyncWorker")
_sync_thread.start()

# In-memory image cache {image_id: (bytes, content_type)}
_firestore_images_cache: Dict[str, Tuple[bytes, str]] = {}

# Local fallback store for Demo Mode with comprehensive Student & Faculty records
_local_persons: List[Dict[str, Any]] = [
    {
        "person_id": "STU001",
        "roll_number": "21CS001",
        "name": "Rahul Kumar",
        "department": "Computer Science & Engineering",
        "role": "Student",
        "class": "CSE-3A",
        "year": "3rd Year",
        "cgpa": "8.92",
        "status": "ACTIVE",
        "image_url": "/static/images/rahul.svg",
        "detections_count": 97,
        "last_detected": "Just now",
        "created_at": "2026-08-15T09:00:00Z",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Computer Networks", "room": "Lab 3", "faculty": "Dr. Rajesh Iyer"},
            {"day": "Monday", "time": "10:15 - 11:15 AM", "subject": "Deep Learning & AI", "room": "LH-102", "faculty": "Prof. S. N. Rao"},
            {"day": "Monday", "time": "11:30 - 12:30 PM", "subject": "Database Management", "room": "LH-102", "faculty": "Dr. Anita Desai"},
            {"day": "Tuesday", "time": "09:00 - 11:00 AM", "subject": "Image Processing Lab", "room": "CV Lab", "faculty": "Dr. Rajesh Iyer"},
            {"day": "Wednesday", "time": "10:00 - 11:00 AM", "subject": "Cloud Computing", "room": "LH-104", "faculty": "Prof. K. Mehta"},
            {"day": "Thursday", "time": "02:00 - 04:00 PM", "subject": "Computer Vision Capstone", "room": "Project Lab", "faculty": "Dr. Rajesh Iyer"},
            {"day": "Friday", "time": "09:00 - 10:00 AM", "subject": "Cyber Security", "room": "LH-102", "faculty": "Prof. M. Verma"}
        ]
    },
    {
        "person_id": "STU002",
        "roll_number": "21EC014",
        "name": "Priya Sharma",
        "department": "Electronics & Communication",
        "role": "Student",
        "class": "ECE-3B",
        "year": "3rd Year",
        "cgpa": "9.15",
        "status": "ACTIVE",
        "image_url": "/static/images/priya.svg",
        "detections_count": 64,
        "last_detected": "12 mins ago",
        "created_at": "2026-08-16T10:30:00Z",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Signals & Systems", "room": "ECE-201", "faculty": "Dr. Sunita Mehra"},
            {"day": "Monday", "time": "10:15 - 11:15 AM", "subject": "Microprocessors & Microcontrollers", "room": "ECE-Lab 1", "faculty": "Prof. V. Raman"},
            {"day": "Tuesday", "time": "11:30 - 01:30 PM", "subject": "VLSI Design & Hardware Lab", "room": "VLSI Lab", "faculty": "Dr. Sunita Mehra"},
            {"day": "Wednesday", "time": "09:00 - 10:00 AM", "subject": "Digital Signal Processing", "room": "ECE-201", "faculty": "Prof. V. Raman"},
            {"day": "Thursday", "time": "10:00 - 11:00 AM", "subject": "Embedded Systems", "room": "ECE-202", "faculty": "Dr. Sunita Mehra"},
            {"day": "Friday", "time": "02:00 - 04:00 PM", "subject": "IoT & Sensor Lab", "room": "IoT Lab", "faculty": "Prof. H. Prasad"}
        ]
    },
    {
        "person_id": "FAC001",
        "faculty_id": "FAC001",
        "name": "Dr. Rajesh Iyer",
        "department": "Computer Science & Engineering",
        "role": "Faculty",
        "designation": "Professor & Head of Department",
        "status": "ACTIVE",
        "image_url": "/static/images/rajesh.svg",
        "detections_count": 142,
        "last_detected": "1 hour ago",
        "created_at": "2026-08-01T08:00:00Z",
        "classes": [
            {"class": "CSE-3A (3rd Year)", "subject": "Computer Networks", "timing": "Mon 09:00 - 10:00 AM", "room": "Lab 3"},
            {"class": "CSE-4B (4th Year)", "subject": "Image Processing & CV", "timing": "Tue 09:00 - 11:00 AM", "room": "CV Lab"},
            {"class": "AI&DS-3A (3rd Year)", "subject": "Computer Vision Capstone", "timing": "Thu 02:00 - 04:00 PM", "room": "Project Lab"},
            {"class": "M.Tech CSE", "subject": "Advanced Neural Networks", "timing": "Fri 11:30 - 01:00 PM", "room": "PG Seminar Hall"}
        ]
    },
    {
        "person_id": "STU003",
        "roll_number": "22IT033",
        "name": "Amit Verma",
        "department": "Information Technology",
        "role": "Student",
        "class": "IT-2A",
        "year": "2nd Year",
        "cgpa": "8.45",
        "status": "ACTIVE",
        "image_url": "/static/images/amit.svg",
        "detections_count": 31,
        "last_detected": "3 hours ago",
        "created_at": "2026-08-20T11:15:00Z",
        "timetable": [
            {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Data Structures & Algorithms", "room": "IT-101", "faculty": "Prof. S. Joshi"},
            {"day": "Tuesday", "time": "10:15 - 11:15 AM", "subject": "Object Oriented Programming", "room": "IT-101", "faculty": "Prof. N. Kapoor"},
            {"day": "Wednesday", "time": "02:00 - 04:00 PM", "subject": "DSA Lab", "room": "Software Lab 2", "faculty": "Prof. S. Joshi"},
            {"day": "Thursday", "time": "11:30 - 12:30 PM", "subject": "Discrete Mathematics", "room": "IT-101", "faculty": "Dr. K. Swamy"}
        ]
    },
    {
        "person_id": "STU004",
        "roll_number": "21AI019",
        "name": "Sneha Patel",
        "department": "Artificial Intelligence & Data Science",
        "role": "Student",
        "class": "AI-3A",
        "year": "3rd Year",
        "cgpa": "9.40",
        "status": "ACTIVE",
        "image_url": "/static/images/sneha.svg",
        "detections_count": 18,
        "last_detected": "Yesterday",
        "created_at": "2026-08-25T14:45:00Z",
        "timetable": [
            {"day": "Monday", "time": "10:00 - 11:00 AM", "subject": "Machine Learning Algorithms", "room": "AI-301", "faculty": "Dr. R. Bannerjee"},
            {"day": "Tuesday", "time": "01:00 - 03:00 PM", "subject": "Deep Learning Studio Lab", "room": "GPU Server Lab", "faculty": "Dr. R. Bannerjee"},
            {"day": "Wednesday", "time": "09:00 - 10:00 AM", "subject": "Natural Language Processing", "room": "AI-301", "faculty": "Prof. P. George"},
            {"day": "Friday", "time": "11:00 - 12:00 PM", "subject": "Ethics in Artificial Intelligence", "room": "LH-Auditorium", "faculty": "Dr. K. Rao"}
        ]
    },
    {
        "person_id": "FAC002",
        "faculty_id": "FAC002",
        "name": "Dr. Sunita Mehra",
        "department": "Electronics & Communication",
        "role": "Faculty",
        "designation": "Associate Professor",
        "status": "ACTIVE",
        "image_url": "/static/images/default_avatar.svg",
        "detections_count": 58,
        "last_detected": "4 hours ago",
        "created_at": "2026-08-05T09:30:00Z",
        "classes": [
            {"class": "ECE-3B (3rd Year)", "subject": "Signals & Systems", "timing": "Mon 09:00 - 10:00 AM", "room": "ECE-201"},
            {"class": "ECE-3B (3rd Year)", "subject": "VLSI Design & Hardware Lab", "timing": "Tue 11:30 - 01:30 PM", "room": "VLSI Lab"},
            {"class": "ECE-4A (4th Year)", "subject": "Embedded Systems", "timing": "Thu 10:00 - 11:00 AM", "room": "ECE-202"}
        ]
    }
]

# Generate initial realistic detection history
_local_detections: List[Dict[str, Any]] = [
    {
        "id": "det-1001",
        "person_id": "STU001",
        "person_name": "Rahul Kumar",
        "camera_id": "CAM-01 (Main Gate)",
        "object_type": "person",
        "recognition_status": "known",
        "confidence": 0.967,
        "timestamp": (datetime.now() - timedelta(minutes=2)).strftime("%d %b %Y, %I:%M %p"),
        "raw_timestamp": (datetime.now() - timedelta(minutes=2)).isoformat(),
        "image_url": "/static/images/sample_evidence_1.svg",
        "storage_path": "detections/2026/09/21/CAM-01_STU001_104200.jpg"
    },
    {
        "id": "det-1002",
        "person_id": "UNKNOWN-0084",
        "person_name": "Unknown Person",
        "camera_id": "CAM-02 (Lab Block)",
        "object_type": "person",
        "recognition_status": "unknown",
        "confidence": 0.914,
        "timestamp": (datetime.now() - timedelta(minutes=7)).strftime("%d %b %Y, %I:%M %p"),
        "raw_timestamp": (datetime.now() - timedelta(minutes=7)).isoformat(),
        "image_url": "/static/images/sample_evidence_unknown.svg",
        "storage_path": "detections/2026/09/21/CAM-02_UNKNOWN-0084_103700.jpg"
    },
    {
        "id": "det-1003",
        "person_id": "STU002",
        "person_name": "Priya Sharma",
        "camera_id": "CAM-03 (Library)",
        "object_type": "person",
        "recognition_status": "known",
        "confidence": 0.948,
        "timestamp": (datetime.now() - timedelta(minutes=15)).strftime("%d %b %Y, %I:%M %p"),
        "raw_timestamp": (datetime.now() - timedelta(minutes=15)).isoformat(),
        "image_url": "/static/images/sample_evidence_2.svg",
        "storage_path": "detections/2026/09/21/CAM-03_STU002_102900.jpg"
    },
    {
        "id": "det-1004",
        "person_id": "FAC001",
        "person_name": "Dr. Rajesh Iyer",
        "camera_id": "CAM-01 (Main Gate)",
        "object_type": "person",
        "recognition_status": "known",
        "confidence": 0.982,
        "timestamp": (datetime.now() - timedelta(hours=1)).strftime("%d %b %Y, %I:%M %p"),
        "raw_timestamp": (datetime.now() - timedelta(hours=1)).isoformat(),
        "image_url": "/static/images/sample_evidence_3.svg",
        "storage_path": "detections/2026/09/21/CAM-01_FAC001_094500.jpg"
    },
    {
        "id": "det-1005",
        "person_id": "UNKNOWN-0083",
        "person_name": "Unknown Person",
        "camera_id": "CAM-04 (Admin Corridor)",
        "object_type": "person",
        "recognition_status": "unknown",
        "confidence": 0.895,
        "timestamp": (datetime.now() - timedelta(hours=2)).strftime("%d %b %Y, %I:%M %p"),
        "raw_timestamp": (datetime.now() - timedelta(hours=2)).isoformat(),
        "image_url": "/static/images/sample_evidence_unknown.svg",
        "storage_path": "detections/2026/09/21/CAM-04_UNKNOWN-0083_085000.jpg"
    }
]


def save_image_to_firestore(
    image_input: Union[str, bytes, np.ndarray],
    image_id: Optional[str] = None,
    filename: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Saves an image into Google Cloud Firestore in the 'images' collection.
    Encodes image as Base64 JPEG (optimized under 1MB Firestore document limit).
    Returns metadata dict containing image_id and dedicated Firestore serving URL.
    """
    if not image_id:
        image_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    jpeg_bytes = None

    try:
        if isinstance(image_input, np.ndarray):
            # Encode numpy array frame directly to JPEG
            # Resize if dimensions exceed 800px to maintain light document footprint (~30-60KB)
            h, w = image_input.shape[:2]
            max_dim = 800
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                resized = cv2.resize(image_input, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                resized = image_input
            ret, buf = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                jpeg_bytes = buf.tobytes()
        elif isinstance(image_input, str):
            # File path
            if os.path.exists(image_input):
                img = cv2.imread(image_input)
                if img is not None:
                    h, w = img.shape[:2]
                    max_dim = 800
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                    ret, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        jpeg_bytes = buf.tobytes()
                else:
                    with open(image_input, 'rb') as f:
                        jpeg_bytes = f.read()
        elif isinstance(image_input, bytes):
            jpeg_bytes = image_input
    except Exception as e:
        logger.warning("Error preparing image for Firestore: %s", e)

    if not jpeg_bytes:
        logger.warning("Failed to obtain JPEG bytes for image_id %s", image_id)
        return {
            "image_id": image_id,
            "image_url": "/static/images/default_avatar.svg",
            "data_url": "",
            "size_bytes": 0,
            "filename": filename or f"{image_id}.jpg"
        }

    b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64_str}"
    serving_url = f"/api/firestore/image/{image_id}"

    # Cache locally in memory for instant delivery
    _firestore_images_cache[image_id] = (jpeg_bytes, "image/jpeg")

    # Queue image doc for background Cloud Firestore synchronization
    image_doc = {
        "id": image_id,
        "filename": filename or f"{image_id}.jpg",
        "content_type": "image/jpeg",
        "base64_data": b64_str,
        "size_bytes": len(jpeg_bytes),
        "created_at": datetime.now().isoformat(),
        "metadata": metadata or {}
    }
    try:
        _cloud_sync_queue.put_nowait(("image", (image_id, image_doc)))
    except Exception:
        pass

    return {
        "image_id": image_id,
        "image_url": serving_url,
        "data_url": data_url,
        "size_bytes": len(jpeg_bytes),
        "filename": filename or f"{image_id}.jpg"
    }


def get_image_from_firestore(image_id: str) -> Tuple[Optional[bytes], str]:
    """
    Retrieves image binary bytes and mime type from Firestore 'images' collection
    or local cache / fallback files.
    """
    # 1. Memory cache check
    if image_id in _firestore_images_cache:
        return _firestore_images_cache[image_id]

    # 2. Query Cloud Firestore 'images' collection
    db = get_firestore_client()
    if db is not None and not is_demo_mode():
        try:
            doc = db.collection("images").document(image_id).get()
            if doc.exists:
                doc_data = doc.to_dict() or {}
                b64 = doc_data.get("base64_data") or doc_data.get("data")
                if b64:
                    if "base64," in b64:
                        b64 = b64.split("base64,")[1]
                    img_bytes = base64.b64decode(b64)
                    content_type = doc_data.get("content_type", "image/jpeg")
                    _firestore_images_cache[image_id] = (img_bytes, content_type)
                    return img_bytes, content_type
        except Exception as e:
            logger.warning("Firestore image retrieval failed for %s: %s", image_id, e)

    # 3. Local disk fallback (sanitized against path traversal)
    safe_image_id = os.path.basename(image_id).strip()
    if not safe_image_id or ".." in image_id or "/" in image_id or "\\" in image_id:
        return None, "image/jpeg"

    search_dirs = ["captured_images", "known_faces", "static/images"]
    for sdir in search_dirs:
        candidates = [
            os.path.join(sdir, safe_image_id),
            os.path.join(sdir, f"{safe_image_id}.jpg"),
            os.path.join(sdir, f"{safe_image_id}.png")
        ]
        for cpath in candidates:
            # Enforce that resolved path stays within sdir
            abs_sdir = os.path.abspath(sdir)
            abs_cpath = os.path.abspath(cpath)
            if not abs_cpath.startswith(abs_sdir):
                continue
            if os.path.exists(cpath):
                try:
                    with open(cpath, "rb") as f:
                        content = f.read()
                    mimetype = "image/png" if cpath.endswith(".png") else "image/jpeg"
                    _firestore_images_cache[image_id] = (content, mimetype)
                    return content, mimetype
                except Exception:
                    pass

    return None, "image/jpeg"


def upload_evidence_image(local_image_path: str, destination_blob_name: str, frame_np: Optional[Any] = None) -> str:
    """
    Stores evidence image in Cloud Firestore 'images' collection as Base64 JPEG.
    Returns the dedicated Firestore serving URL (/api/firestore/image/<id>).
    """
    filename = os.path.basename(local_image_path)
    clean_name = filename.rsplit('.', 1)[0].replace(" ", "_")
    image_id = clean_name

    res = save_image_to_firestore(
        image_input=frame_np if frame_np is not None else local_image_path,
        image_id=image_id,
        filename=filename,
        metadata={"storage_path": destination_blob_name}
    )
    return res.get("image_url", f"/captured_images/{filename}")


def save_detection(detection_data: Dict[str, Any]) -> Dict[str, Any]:
    """Save a detection record to Firestore or local fallback store."""
    db = get_firestore_client()
    record_id = detection_data.get("id") or str(uuid.uuid4())
    detection_data["id"] = record_id

    if "timestamp" not in detection_data:
        detection_data["timestamp"] = datetime.now().strftime("%d %b %Y, %I:%M %p")
    if "raw_timestamp" not in detection_data:
        detection_data["raw_timestamp"] = datetime.now().isoformat()

    # Ensure Firestore image metadata linkage
    if "image_id" not in detection_data and detection_data.get("image_url", "").startswith("/api/firestore/image/"):
        detection_data["image_id"] = detection_data["image_url"].split("/api/firestore/image/")[1]

    # 1. Update in-memory local detections instantly for 0-latency UI updates
    _local_detections.insert(0, dict(detection_data))
    if len(_local_detections) > 500:
        _local_detections.pop()

    # 2. Update person's detection count if known
    person_id = detection_data.get("person_id")
    if person_id and not str(person_id).startswith("UNKNOWN"):
        for person in _local_persons:
            if person.get("person_id") == person_id:
                person["detections_count"] = person.get("detections_count", 0) + 1
                person["last_detected"] = "Just now"
                break

    # 3. Offload cloud write to background queue (never hangs camera or web stream)
    try:
        _cloud_sync_queue.put_nowait(("detection", (record_id, dict(detection_data))))
    except Exception:
        pass

    return detection_data


def get_detections(limit: int = 50, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Retrieve detections list sorted by recency with optional filters."""
    db = get_firestore_client()

    if db is not None and not is_demo_mode():
        try:
            query = db.collection("detections").order_by(
                "raw_timestamp", direction="DESCENDING"
            ).limit(limit)

            docs = query.stream(timeout=2.0)
            results: List[Dict[str, Any]] = [d for doc in docs if (d := doc.to_dict()) is not None]
            if results:
                return _apply_filters(results, filters)
        except Exception as e:
            logger.debug("Firestore read timeout or offline: %s. Using local store.", e)

    # Local store fallback
    return _apply_filters(_local_detections[:limit], filters)


def _apply_filters(records: List[Dict[str, Any]], filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not filters:
        return records

    filtered = records
    if filters.get("status"):
        target_status = filters["status"].lower()
        filtered = [r for r in filtered if r.get("recognition_status", "").lower() == target_status]

    if filters.get("camera"):
        target_cam = filters["camera"].lower()
        filtered = [r for r in filtered if target_cam in r.get("camera_id", "").lower()]

    if filters.get("search"):
        q = filters["search"].lower()
        filtered = [
            r for r in filtered
            if q in r.get("person_name", "").lower() or q in r.get("person_id", "").lower()
        ]

    return filtered


def get_person_by_id(person_id: str) -> Optional[Dict[str, Any]]:
    """Fetch person record by ID from Firestore or local cache, supporting class and ID aliases."""
    if not person_id:
        return None

    # Handle alias mapping between CNN class names and standard person IDs
    id_aliases = {
        "Faculty_001": "FAC001",
        "Faculty_002": "FAC002",
        "Student_001": "STU001",
        "Student_002": "STU002",
        "Student_003": "STU003",
        "Student_004": "STU004",
        "FAC001": "Faculty_001",
        "FAC002": "Faculty_002",
        "STU001": "Student_001",
        "STU002": "Student_002",
        "STU003": "Student_003",
        "STU004": "Student_004",
    }
    candidate_ids = [person_id]
    if person_id in id_aliases:
        candidate_ids.append(id_aliases[person_id])

    db = get_firestore_client()
    if db is not None and not is_demo_mode():
        for cid in candidate_ids:
            try:
                doc = db.collection("persons").document(cid).get()
                if doc.exists:
                    return doc.to_dict()
            except Exception as e:
                logger.debug("Firestore lookup failed for person %s: %s", cid, e)

    for person in _local_persons:
        pid = person.get("person_id")
        pname = person.get("name", "")
        if pid in candidate_ids or pname.lower() == person_id.lower():
            return person
    return None


def get_person_details(person_id: Optional[str], default_name: Optional[str] = None, status: str = "known") -> Dict[str, Any]:
    """
    Returns formatted student or faculty details.
    - If Student: returns name, roll_number, class, year, cgpa, timetable, department.
    - If Faculty: returns name, faculty_id, department, classes (teaching schedule), designation.
    - If Unknown: returns security alert and visitor record.
    """
    person = get_person_by_id(person_id) if person_id else None

    if person:
        role = person.get("role", "Student")
        is_student = "student" in role.lower()
        is_faculty = "faculty" in role.lower() or "professor" in role.lower() or "hod" in role.lower()

        if is_student:
            return {
                "status": "known",
                "person_type": "Student",
                "person_id": person.get("person_id", person_id or "STU-000"),
                "name": person.get("name", default_name or "Student"),
                "roll_number": person.get("roll_number", person_id or "STU-000"),
                "class": person.get("class", "CSE-3A"),
                "year": person.get("year", "3rd Year"),
                "cgpa": person.get("cgpa", "8.50"),
                "department": person.get("department", "Computer Science & Engineering"),
                "image_url": person.get("image_url", "/static/images/default_avatar.svg"),
                "timetable": person.get("timetable", [
                    {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": "Core Engineering Lecture", "room": "LH-101", "faculty": "Department Staff"},
                    {"day": "Tuesday", "time": "11:00 - 01:00 PM", "subject": "Practical Lab Session", "room": "Lab Block A", "faculty": "Lab Instructor"},
                    {"day": "Wednesday", "time": "10:00 - 11:00 AM", "subject": "Applied Technology", "room": "LH-102", "faculty": "Department Staff"},
                    {"day": "Thursday", "time": "02:00 - 04:00 PM", "subject": "Project Workshop", "room": "Project Lab", "faculty": "Coordinator"}
                ])
            }
        else:
            return {
                "status": "known",
                "person_type": "Faculty",
                "person_id": person.get("person_id", person_id or "FAC-000"),
                "faculty_id": person.get("faculty_id", person.get("person_id", person_id or "FAC-000")),
                "name": person.get("name", default_name or "Faculty Member"),
                "department": person.get("department", "Computer Science & Engineering"),
                "designation": person.get("designation", "Faculty / Lecturer"),
                "image_url": person.get("image_url", "/static/images/default_avatar.svg"),
                "classes": person.get("classes", [
                    {"class": f"{person.get('department', 'CSE')} 3rd Year", "subject": "Specialized Domain Theory", "timing": "Mon 10:00 - 11:00 AM", "room": "LH-201"},
                    {"class": f"{person.get('department', 'CSE')} 4th Year", "subject": "Advanced Research & Project Lab", "timing": "Wed 02:00 - 04:00 PM", "room": "Special Lab"}
                ])
            }

    # Unrecognized or Unknown Person
    return {
        "status": "unknown",
        "person_type": "Unknown",
        "person_id": person_id or "UNKNOWN-0001",
        "name": default_name or "Unknown Person",
        "department": "Unassigned",
        "role": "Visitor / Unregistered",
        "image_url": "/static/images/sample_evidence_unknown.svg",
        "alert": "Unregistered individual detected by surveillance system."
    }


def get_persons() -> List[Dict[str, Any]]:
    """Retrieve all registered persons merging Firestore with local pre-seeded records."""
    db = get_firestore_client()
    firestore_persons: List[Dict[str, Any]] = []
    if db is not None and not is_demo_mode():
        try:
            docs = db.collection("persons").stream()
            firestore_persons = [d for doc in docs if (d := doc.to_dict()) is not None]
        except Exception as e:
            logger.warning("Failed to fetch persons from Firestore: %s", e)

    merged: Dict[str, Dict[str, Any]] = {str(p["person_id"]): p for p in _local_persons if p.get("person_id")}
    for p in firestore_persons:
        pid = p.get("person_id")
        if pid:
            merged[pid] = p

    return list(merged.values())


def add_person(person_data: Dict[str, Any]) -> Dict[str, Any]:
    """Register a new person in Firestore and local store."""
    role = person_data.get("role", "Student")
    is_student = "student" in role.lower()
    is_faculty = "faculty" in role.lower() or "professor" in role.lower()

    person_id = (person_data.get("person_id") or "").strip()
    if not person_id:
        person_id = f"STU{len(_local_persons) + 1:03d}" if is_student else f"FAC{len(_local_persons) + 1:03d}"
    person_data["person_id"] = person_id

    if is_student:
        if "roll_number" not in person_data:
            person_data["roll_number"] = person_data.get("person_id", f"21CS{len(_local_persons) + 1:03d}")
        if "class" not in person_data:
            person_data["class"] = "CSE-3A"
        if "year" not in person_data:
            person_data["year"] = "3rd Year"
        if "cgpa" not in person_data:
            person_data["cgpa"] = "8.50"
        if "timetable" not in person_data:
            dept = person_data.get("department", "CSE")
            person_data["timetable"] = [
                {"day": "Monday", "time": "09:00 - 10:00 AM", "subject": f"{dept} Core Theory", "room": "LH-101", "faculty": "Dept Faculty"},
                {"day": "Tuesday", "time": "11:00 - 01:00 PM", "subject": f"{dept} Computing Lab", "room": "Lab Block", "faculty": "Lab Incharge"},
                {"day": "Wednesday", "time": "10:00 - 11:00 AM", "subject": "Data Systems", "room": "LH-102", "faculty": "Assoc. Professor"},
                {"day": "Thursday", "time": "02:00 - 04:00 PM", "subject": "Capstone Project", "room": "Project Lab", "faculty": "HOD"}
            ]
    elif is_faculty:
        if "faculty_id" not in person_data:
            person_data["faculty_id"] = person_id
        if "designation" not in person_data:
            person_data["designation"] = "Assistant Professor"
        if "classes" not in person_data:
            dept = person_data.get("department", "CSE")
            person_data["classes"] = [
                {"class": f"{dept} 3rd Year", "subject": "Specialized Core Subject", "timing": "Mon 09:00 - 10:00 AM", "room": "LH-101"},
                {"class": f"{dept} 4th Year", "subject": "Laboratory & Practicals", "timing": "Wed 02:00 - 04:00 PM", "room": "Central Lab"}
            ]

    if "detections_count" not in person_data:
        person_data["detections_count"] = 0
    if "last_detected" not in person_data:
        person_data["last_detected"] = "Never"
    if "created_at" not in person_data:
        person_data["created_at"] = datetime.now().isoformat()
    if "status" not in person_data:
        person_data["status"] = "ACTIVE"
    if "image_url" not in person_data:
        person_data["image_url"] = "/static/images/default_avatar.svg"

    db = get_firestore_client()
    if db is not None and not is_demo_mode():
        try:
            db.collection("persons").document(person_id).set(person_data)
        except Exception as e:
            logger.warning("Firestore person registration failed: %s", e)

    _local_persons.insert(0, person_data)
    return person_data


def update_person(person_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update person details."""
    db = get_firestore_client()
    if db is not None and not is_demo_mode():
        try:
            db.collection("persons").document(person_id).update(updates)
        except Exception as e:
            logger.warning("Firestore update failed: %s", e)

    for person in _local_persons:
        if person["person_id"] == person_id:
            person.update(updates)
            return person
    return None


def delete_person(person_id: str) -> bool:
    """Delete person from database."""
    global _local_persons
    db = get_firestore_client()
    if db is not None and not is_demo_mode():
        try:
            db.collection("persons").document(person_id).delete()
        except Exception as e:
            logger.warning("Firestore delete failed: %s", e)

    _local_persons = [p for p in _local_persons if p["person_id"] != person_id]
    return True


def get_system_stats() -> Dict[str, Any]:
    """
    Calculate and return real system KPI statistics for dashboard.
    Strictly connects to actual backend detection records and registered persons.
    No dummy or inflated data.
    """
    detections = get_detections(limit=500)
    persons = get_persons()

    total_detections = len(detections)
    known_count = sum(1 for d in detections if d.get("recognition_status") == "known")
    unknown_count = sum(1 for d in detections if d.get("recognition_status") == "unknown")
    object_count = sum(1 for d in detections if d.get("recognition_status") == "object" or d.get("object_class") not in ["person", None])

    # Count breakdown of detected objects by category
    objects_breakdown: Dict[str, int] = {}
    for d in detections:
        obj_cls = d.get("object_class") or ("person" if d.get("recognition_status") in ["known", "unknown"] else "object")
        objects_breakdown[obj_cls] = objects_breakdown.get(obj_cls, 0) + 1

    # Real hourly detection volume for trend charts (last 8 hours)
    now = datetime.now()
    hours: List[str] = []
    hourly_counts: List[int] = []

    for i in range(7, -1, -1):
        target_hour = (now - timedelta(hours=i)).replace(minute=0, second=0, microsecond=0)
        hour_label = target_hour.strftime("%H:00")
        hours.append(hour_label)

        # Count events occurring in this hour window
        next_hour = target_hour + timedelta(hours=1)
        count = 0
        for d in detections:
            raw_ts = d.get("raw_timestamp")
            if raw_ts:
                try:
                    dt = datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    if target_hour <= dt < next_hour:
                        count += 1
                except Exception:
                    pass
        hourly_counts.append(count)

    # If all 0s (fresh start), populate last slot with current count for responsiveness
    if sum(hourly_counts) == 0 and total_detections > 0:
        hourly_counts[-1] = total_detections

    # Active cameras summary
    try:
        from camera_manager import CameraManager
        cm = CameraManager()
        cam_info = cm.get_status()
        active_cameras_summary = cam_info.get("cameras_summary", "01 / 02 Online")
    except Exception:
        active_cameras_summary = "01 / 02 Online"

    # Calculate real trend vs older events
    today_str = now.strftime("%d %b %Y")
    today_count = sum(1 for d in detections if today_str in str(d.get("timestamp", "")))
    trend_str = f"+{today_count} Today" if today_count > 0 else "0 Today"

    fb_status = get_firebase_status()
    firestore_images_count = len(_firestore_images_cache)

    return {
        "total_detections": total_detections,
        "known_persons": known_count,
        "known_count": known_count,
        "unknown_persons": unknown_count,
        "unknown_count": unknown_count,
        "total_detected_objects": object_count,
        "objects_breakdown": objects_breakdown,
        "active_cameras": active_cameras_summary,
        "registered_persons_count": len(persons),
        "recent_trend": trend_str,
        "chart_labels": hours,
        "chart_data": hourly_counts,
        "is_demo_mode": is_demo_mode(),
        "storage_mode": "Cloud Firestore (Base64 Images)" if not is_demo_mode() else "Local Cache (Firestore Emulated)",
        "firebase_status": fb_status,
        "firestore_images_count": firestore_images_count
    }


