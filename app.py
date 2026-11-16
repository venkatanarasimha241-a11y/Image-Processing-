"""
app.py
------
AI Camera Monitoring System - Main Flask Application Server.
Provides web dashboard templates, live video streaming, and REST APIs
for persons management, detection logs, analytics, RTSP CCTV controls, and thresholds.
"""

import os
import uuid
import logging
from datetime import datetime
import cv2
import numpy as np

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from flask import (
    Flask, render_template, Response, request,
    jsonify, redirect, url_for, session, send_from_directory
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("App")

# Import system modules
import firebase_config
import firebase_service
from camera_manager import CameraManager, generate_mjpeg_stream
from rtsp_camera import RTSPCamera, mask_rtsp_url
from recognition import KNOWN_FACES_DIR
from config import CAPTURED_IMAGES_DIR, KNOWN_FACES_DIR

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai-camera-monitoring-secret-key-2026")

# Ensure required directories exist
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize Firebase SDK
firebase_config.initialize_firebase()

# Global camera manager instance
camera_manager = CameraManager()


# ---------------------------------------------------------
# Static File & Upload Serving
# ---------------------------------------------------------

@app.route("/captured_images/<path:filename>")
def serve_captured_image(filename):
    """Serves locally stored evidence images."""
    return send_from_directory(CAPTURED_IMAGES_DIR, filename)


@app.route("/known_faces/<path:filename>")
def serve_known_face(filename):
    """Serves registered person face images."""
    return send_from_directory(KNOWN_FACES_DIR, filename)


@app.route("/api/firestore/image/<image_id>")
def serve_firestore_image(image_id):
    """
    Streams binary image data directly from Cloud Firestore 'images' collection.
    Decodes stored Base64 JPEG with caching headers and local fallback.
    """
    try:
        img_bytes, content_type = firebase_service.get_image_from_firestore(image_id)
        if img_bytes:
            return Response(
                img_bytes,
                mimetype=content_type,
                headers={
                    "Cache-Control": "public, max-age=86400",
                    "Content-Disposition": f"inline; filename={image_id}.jpg"
                }
            )
    except Exception as e:
        logger.warning("Error serving Firestore image %s: %s", image_id, e)

    return redirect(url_for('static', filename='images/default_avatar.svg'))


# ---------------------------------------------------------
# Web Page Routes
# ---------------------------------------------------------

@app.route("/")
def index():
    """Main AI Monitoring Dashboard."""
    user = session.get("user", {
        "name": "Security Administrator",
        "email": "admin@aicamera.com",
        "role": "System Administrator",
        "model": "admin",
        "department": "All Departments"
    })
    stats = firebase_service.get_system_stats()
    cam_info = camera_manager.get_status()
    stats["camera_name"] = cam_info.get("camera", {}).get("name", "Main Gate CCTV")
    stats["camera_location"] = cam_info.get("camera", {}).get("location", "Main Gate")
    stats["camera_fps"] = cam_info.get("fps", 0.0)
    stats["active_cameras"] = cam_info.get("cameras_summary", "01 / 02 Online")
    return render_template("index.html", user=user, stats=stats)


@app.route("/login")
def login_page():
    """Login authentication page."""
    return render_template("login.html")


@app.route("/persons")
def persons_page():
    """Registered Persons management interface."""
    user = session.get("user", {
        "name": "Security Administrator",
        "email": "admin@aicamera.com",
        "role": "System Administrator",
        "model": "admin",
        "department": "All Departments"
    })
    persons = firebase_service.get_persons()

    # Department scoped view if logged in under department model
    if user.get("model") == "department" and user.get("department"):
        dept_name = user["department"].lower()
        dept_persons = [p for p in persons if dept_name in p.get("department", "").lower()]
        if dept_persons:
            persons = dept_persons

    return render_template("persons.html", user=user, persons=persons)


@app.route("/detections")
def detections_page():
    """Detection logs and audit trail page."""
    user = session.get("user", {
        "name": "Security Administrator",
        "email": "admin@aicamera.com",
        "role": "System Administrator",
        "model": "admin",
        "department": "All Departments"
    })
    detections = firebase_service.get_detections(limit=100)
    return render_template("detections.html", user=user, detections=detections)


@app.route("/logout")
def logout():
    """Clears user session and redirects to login."""
    session.clear()
    return redirect(url_for("login_page"))


# ---------------------------------------------------------
# Video Streaming Route
# ---------------------------------------------------------

@app.route("/video_feed")
@app.route("/video_feed/<camera_id>")
def video_feed(camera_id=None):
    """Live multipart MJPEG video stream with YOLO and Face Recognition overlays."""
    if camera_id and camera_id in camera_manager.cameras:
        camera_manager.switch_camera(camera_id)

    return Response(
        generate_mjpeg_stream(camera_manager),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------
# REST API Endpoints
# ---------------------------------------------------------

@app.route("/api/login", methods=["POST"])
def api_login():
    """Handles authentication for Admin and Department models."""
    try:
        data = request.get_json(silent=True) or request.form
        login_model = data.get("model", "admin").strip().lower()
        email = (data.get("email") or data.get("username") or "").strip()
        password = data.get("password", "").strip()
        department = data.get("department", "Computer Science & Engineering").strip()

        if not email or not password:
            return jsonify({"success": False, "error": "Email and password are required"}), 400

        if login_model in ["department", "departments"]:
            dept_short = "".join(w[0] for w in department.split() if w[0].isupper()) or "DEPT"
            user_info = {
                "name": f"{dept_short} Department Coordinator",
                "email": email,
                "role": "Department Portal",
                "model": "department",
                "department": department
            }
        else:
            user_info = {
                "name": "Security Administrator" if "admin" in email.lower() else email.split("@")[0].capitalize(),
                "email": email,
                "role": "System Administrator",
                "model": "admin",
                "department": "All Departments"
            }

        session["user"] = user_info

        return jsonify({
            "success": True,
            "message": f"Authenticated successfully as {user_info['role']}",
            "user": user_info
        }), 200

    except Exception as e:
        logger.error("Login API error: %s", e)
        return jsonify({"success": False, "error": "Internal authentication error"}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns real-time KPI statistics and chart metrics from actual backend data."""
    try:
        stats = firebase_service.get_system_stats()
        cam_status = camera_manager.get_status()
        stats["camera_fps"] = cam_status.get("fps", 0.0)
        stats["active_cameras"] = cam_status.get("cameras_summary", "01 / 02 Online")
        active_cam = cam_status.get("camera", {})
        stats["camera_id"] = active_cam.get("camera_id", camera_manager.active_camera_id)
        stats["camera_name"] = active_cam.get("name", "CCTV Camera")
        stats["camera_location"] = active_cam.get("location", "Main Gate")
        stats["connection_status"] = active_cam.get("connection_status", "ONLINE")
        stats["rtsp_url_masked"] = active_cam.get("masked_url", "")
        return jsonify({"success": True, "stats": stats, "camera": cam_status}), 200
    except Exception as e:
        logger.error("Stats API error: %s", e)
        return jsonify({"success": False, "error": "Failed to fetch stats"}), 500


@app.route("/api/detections", methods=["GET"])
def api_detections():
    """Returns detection records with optional query parameters."""
    try:
        try:
            limit = int(request.args.get("limit", 50))
        except (ValueError, TypeError):
            return jsonify({"success": False, "error": "Invalid limit parameter"}), 400
        filters = {
            "status": request.args.get("status"),
            "camera": request.args.get("camera"),
            "search": request.args.get("search")
        }
        filters = {k: v for k, v in filters.items() if v}
        records = firebase_service.get_detections(limit=limit, filters=filters)
        return jsonify({"success": True, "count": len(records), "detections": records}), 200
    except Exception as e:
        logger.error("Detections API error: %s", e)
        return jsonify({"success": False, "error": "Failed to fetch detections"}), 500


@app.route("/api/persons", methods=["GET", "POST"])
def api_persons():
    """Get list of registered persons or register a new person."""
    if request.method == "GET":
        try:
            persons = firebase_service.get_persons()
            return jsonify({"success": True, "persons": persons}), 200
        except Exception as e:
            logger.error("Get persons error: %s", e)
            return jsonify({"success": False, "error": "Failed to load persons"}), 500

    # POST: Add new person
    try:
        data = request.get_json(silent=True) or request.form.to_dict()

        name = data.get("name", "").strip()
        department = data.get("department", "").strip()
        role = data.get("role", "Student").strip()
        person_id = data.get("person_id", "").strip()
        is_student = "student" in role.lower()
        if not person_id:
            existing_count = len(firebase_service.get_persons()) + 1
            person_id = f"STU{existing_count:03d}" if is_student else f"FAC{existing_count:03d}"

        roll_number = data.get("roll_number", "").strip()
        class_name = data.get("class", "").strip()
        year = data.get("year", "").strip()
        cgpa = data.get("cgpa", "").strip()
        faculty_id = data.get("faculty_id", "").strip()
        designation = data.get("designation", "").strip()

        if not name:
            return jsonify({"success": False, "error": "Person name is required"}), 400

        # 1. Prevent duplicate person_id registrations
        existing_person = firebase_service.get_person_by_id(person_id)
        if existing_person and existing_person.get("name") and existing_person.get("status") != "unknown":
            return jsonify({"success": False, "error": f"Person ID '{person_id}' is already registered as '{existing_person.get('name')}'"}), 400

        # 2. Handle face photo upload if provided with Phase 3 validation
        image_url = "/static/images/default_avatar.svg"
        saved_embedding_list = None
        if "photo" in request.files:
            file = request.files["photo"]
            if file and file.filename != "":
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img_bgr is None or img_bgr.size == 0:
                    return jsonify({"success": False, "error": "Uploaded photo could not be decoded as a valid image."}), 400

                # Validate and register directly through unified pipeline
                reg_res = camera_manager.recognizer.register_new_face(
                    person_id=person_id,
                    name=name,
                    image_bgr=img_bgr,
                    metadata={"department": department, "role": role}
                )
                if not reg_res.success:
                    return jsonify({"success": False, "error": reg_res.message}), 400

                safe_pid = os.path.basename(str(person_id).strip().replace("/", "_").replace("\\", "_"))
                clean_name = os.path.basename(str(name).strip().replace(" ", "_").replace("/", "_").replace("\\", "_"))
                filename = f"{safe_pid}_{clean_name}.jpg"
                save_path = os.path.join(KNOWN_FACES_DIR, filename)

                # Store face image in Firestore 'images' collection
                img_id = f"person_{person_id}_{int(datetime.now().timestamp())}"
                stored_img = firebase_service.save_image_to_firestore(
                    image_input=save_path,
                    image_id=img_id,
                    filename=filename,
                    metadata={"person_id": person_id, "name": name, "role": role}
                )
                image_url = stored_img.get("image_url", f"/known_faces/{filename}")
                if reg_res.embedding is not None:
                    saved_embedding_list = reg_res.embedding.tolist()

        new_person = {
            "person_id": person_id,
            "name": name,
            "department": department or "Computer Science & Engineering",
            "role": role,
            "image_url": image_url,
            "status": "ACTIVE"
        }
        if saved_embedding_list:
            new_person["embedding"] = saved_embedding_list

        if is_student:
            new_person["roll_number"] = roll_number or person_id
            new_person["class"] = class_name or "CSE-3A"
            new_person["year"] = year or "3rd Year"
            new_person["cgpa"] = cgpa or "8.50"
        elif "faculty" in role.lower():
            new_person["faculty_id"] = faculty_id or person_id
            new_person["designation"] = designation or "Assistant Professor"

        created = firebase_service.add_person(new_person)
        return jsonify({"success": True, "message": "Person registered successfully", "person": created}), 201

    except Exception as e:
        logger.error("Add person error: %s", e)
        return jsonify({"success": False, "error": "Failed to register person"}), 500


@app.route("/api/persons/<person_id>", methods=["GET", "PUT", "DELETE"])
def api_person_detail(person_id):
    """Retrieve, update, or delete a person profile."""
    if request.method == "GET":
        details = firebase_service.get_person_details(person_id)
        if details:
            return jsonify({"success": True, "person": details}), 200
        return jsonify({"success": False, "error": "Person not found"}), 404

    if request.method == "DELETE":
        try:
            success = firebase_service.delete_person(person_id)
            camera_manager.recognizer.load_known_faces()
            return jsonify({"success": success, "message": f"Person {person_id} deleted"}), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # PUT: Update
    try:
        updates = request.get_json(silent=True) or request.form.to_dict()
        updated = firebase_service.update_person(person_id, updates)
        if updated:
            camera_manager.recognizer.load_known_faces()
            return jsonify({"success": True, "person": updated}), 200
        return jsonify({"success": False, "error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------
# RTSP & Camera Controls API
# ---------------------------------------------------------

@app.route("/api/camera/connect", methods=["POST"])
def api_camera_connect():
    """
    Connects to a CCTV / IP Camera stream via RTSP or Webcam.
    Allows administrator to specify RTSP URL, Camera ID, Name, and Location.
    Verifies connection and begins processing.
    """
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        rtsp_url = data.get("rtsp_url", "").strip()
        camera_id = data.get("camera_id", "").strip() or camera_manager.active_camera_id
        name = data.get("name", "CCTV Stream").strip()
        location = data.get("location", "Campus").strip()

        if not rtsp_url:
            return jsonify({"success": False, "error": "RTSP URL or camera device index is required"}), 400

        # Verify stream first
        is_valid, msg, info = RTSPCamera.verify_stream(rtsp_url)
        if not is_valid:
            return jsonify({"success": False, "error": f"Connection check failed: {msg}"}), 400

        # Register and start camera
        cam = camera_manager.add_camera(camera_id, rtsp_url, name, location)
        camera_manager.switch_camera(camera_id)
        cam.start()

        return jsonify({
            "success": True,
            "message": f"Connected to camera [{name}] successfully!",
            "camera_id": camera_id,
            "info": info,
            "status": camera_manager.get_status()
        }), 200

    except Exception as e:
        logger.error("Camera connect error: %s", e)
        return jsonify({"success": False, "error": f"Failed to connect: {str(e)}"}), 500


@app.route("/api/camera/verify", methods=["POST"])
def api_camera_verify():
    """Verifies whether an RTSP URL or IP camera is reachable without switching camera."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        rtsp_url = data.get("rtsp_url", "").strip()
        if not rtsp_url:
            return jsonify({"success": False, "error": "RTSP URL is required"}), 400

        is_valid, msg, info = RTSPCamera.verify_stream(rtsp_url)
        return jsonify({
            "success": is_valid,
            "message": msg,
            "stream_info": info
        }), (200 if is_valid else 400)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    """Start video stream."""
    camera_manager.start()
    return jsonify({"success": True, "message": "Camera started", "status": camera_manager.get_status()}), 200


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    """Stop video stream."""
    camera_manager.stop()
    return jsonify({"success": True, "message": "Camera stopped", "status": camera_manager.get_status()}), 200


@app.route("/api/camera/disconnect", methods=["POST"])
def api_camera_disconnect():
    """Disconnect active camera and release resources."""
    camera_manager.disconnect_active()
    return jsonify({"success": True, "message": "Camera disconnected and resources released", "status": camera_manager.get_status()}), 200


@app.route("/api/camera/status", methods=["GET"])
def api_camera_status():
    """Returns real-time camera connection status, telemetry, and all configured CCTV streams."""
    return jsonify({"success": True, "camera": camera_manager.get_status()}), 200


@app.route("/api/camera/switch", methods=["POST"])
def api_camera_switch():
    """Switches active camera to another registered camera ID."""
    try:
        data = request.get_json(silent=True) or request.form.to_dict()
        cam_key = data.get("camera_id", "").strip()
        if not cam_key:
            return jsonify({"success": False, "error": "Camera ID is required"}), 400

        success = camera_manager.switch_camera(cam_key)
        return jsonify({
            "success": success,
            "message": f"Switched to camera: {cam_key}" if success else f"Failed to switch to camera {cam_key}",
            "camera": camera_manager.get_status()
        }), (200 if success else 400)
    except Exception as e:
        logger.error("Camera switch error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/identify_image", methods=["POST"])
def api_identify_image():
    """
    Receives an uploaded photo (JPG, JPEG, PNG, WEBP), performs complete YOLO object detection
    and deep face recognition, generates an annotated bounding-box image, saves evidence to storage
    and Firestore database, and returns rich detection results.
    """
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided in request"}), 400

        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"success": False, "error": "Empty image file provided"}), 400

        filename: str = file.filename

        # Validate file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
            return jsonify({
                "success": False,
                "error": f"Unsupported image file format '{ext}'. Only JPG, JPEG, PNG, and WEBP images are supported."
            }), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        if len(file_bytes) == 0:
            return jsonify({"success": False, "error": "Uploaded image file is zero bytes"}), 400

        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            return jsonify({"success": False, "error": "Invalid or corrupted image content. Could not decode image."}), 400

        h, w = img.shape[:2]
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_suffix = os.urandom(3).hex()
        image_id = f"upload_{time_str}_{unique_suffix}"

        # 1. Save original image locally
        orig_filename = f"original_{image_id}.jpg"
        orig_path = os.path.join(CAPTURED_IMAGES_DIR, orig_filename)
        cv2.imwrite(orig_path, img)

        orig_img_res = firebase_service.save_image_to_firestore(
            image_input=img,
            image_id=f"orig_{image_id}",
            filename=orig_filename,
            metadata={"type": "uploaded_original", "filename": filename}
        )
        orig_url = orig_img_res.get("image_url", f"/captured_images/{orig_filename}")

        # 2. Run YOLO detection in single-image mode
        yolo_dets = camera_manager.yolo.detect(img, is_single_image=True)
        recognized_entities = []

        for idx, det in enumerate(yolo_dets):
            bbox = det["bbox"]
            conf = det["confidence"]
            cls_name = det["class_name"]

            if det["is_person"]:
                rec = camera_manager.recognizer.recognize(
                    frame=img,
                    person_bbox=bbox,
                    track_id=idx + 1,
                    yolo_conf=conf,
                    is_single_image=True
                )
                rec["bbox"] = bbox
                rec["class_name"] = "person"
                recognized_entities.append(rec)
            else:
                recognized_entities.append({
                    "status": "object",
                    "recognition_status": "object",
                    "class_name": cls_name,
                    "canonical_class": det.get("canonical_class", cls_name),
                    "confidence": conf,
                    "yolo_confidence": conf,
                    "similarity_score": conf,
                    "bbox": bbox,
                    "name": cls_name.title(),
                    "person_id": None,
                    "department": "Electronics" if det.get("is_electronic") else "Objects",
                    "role": "Detected Object"
                })

        # 3. If no YOLO person/object was found, attempt face recognition directly on the image
        if not recognized_entities:
            rec = camera_manager.recognizer.recognize(
                frame=img,
                person_bbox=[0, 0, w, h],
                track_id=1,
                yolo_conf=0.90,
                is_single_image=True
            )
            if rec.get("face_detected"):
                rec["bbox"] = rec.get("face_bbox") or [0, 0, w, h]
                rec["class_name"] = "person"
                recognized_entities.append(rec)

        # If still nothing detected, inform user gracefully
        if not recognized_entities:
            return jsonify({
                "success": True,
                "message": "No recognizable faces or objects were detected in the uploaded photo.",
                "entities_count": 0,
                "results": [],
                "original_image_url": orig_url,
                "annotated_image_url": orig_url,
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M:%S %p")
            }), 200

        # 4. Create annotated result image with bounding boxes & label banners
        annotated_img = img.copy()

        COLOR_KNOWN = (34, 197, 94)       # Emerald Green
        COLOR_UNKNOWN = (239, 68, 68)     # Crimson Red
        COLOR_ELECTRONIC = (249, 115, 22) # Amber / Orange (Mobile phone, Laptop)
        COLOR_VEHICLE = (59, 130, 246)    # Royal Blue
        COLOR_OBJECT = (168, 85, 247)     # Purple

        for ent in recognized_entities:
            bbox = ent.get("bbox")
            if not bbox or len(bbox) < 4:
                continue

            x1, y1, x2, y2 = bbox
            cls_name = ent.get("class_name", "object").lower()
            status = ent.get("status", "unknown").lower()
            name = ent.get("name", cls_name.title())

            if cls_name == "person":
                if status == "known":
                    box_color = COLOR_KNOWN
                    sim = ent.get("similarity_score", 0.0)
                    title = f"{name} ({sim:.0%})"
                    badge = f"KNOWN | {ent.get('person_id', 'ID')}"
                elif status == "unknown":
                    box_color = COLOR_UNKNOWN
                    sim = ent.get("similarity_score", 0.0)
                    title = f"UNKNOWN ({sim:.0%})"
                    badge = "UNREGISTERED VISITOR"
                else:
                    box_color = (148, 163, 184)
                    y_conf = ent.get("yolo_confidence", ent.get("confidence", 0.0))
                    title = f"PERSON ({y_conf:.0%})"
                    badge = "FACE NOT AVAILABLE"
            elif cls_name in ["mobile phone", "cell phone", "laptop", "tv", "electronic"]:
                box_color = COLOR_ELECTRONIC
                conf = ent.get("yolo_confidence", ent.get("confidence", 0.0))
                title = f"{cls_name.upper()} ({conf:.0%})"
                badge = "MOBILE PHONE" if cls_name in ["mobile phone", "cell phone"] else "ELECTRONIC"
            elif cls_name in ["car", "motorcycle", "bus", "truck", "bicycle"]:
                box_color = COLOR_VEHICLE
                conf = ent.get("confidence", 0.0)
                title = f"{cls_name.upper()} ({conf:.0%})"
                badge = "VEHICLE"
            else:
                box_color = COLOR_OBJECT
                conf = ent.get("confidence", 0.0)
                title = f"{cls_name.upper()} ({conf:.0%})"
                badge = "SURVEILLANCE OBJECT"

            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), box_color, 2)

            # Draw face box inside person if present
            if ent.get("face_bbox"):
                fx1, fy1, fx2, fy2 = ent["face_bbox"]
                cv2.rectangle(annotated_img, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)

            # Draw Label Banner
            banner_w = max(140, len(title) * 9 + 20)
            banner_h = 36
            by1 = max(0, y1 - banner_h)
            by2 = y1
            bx1 = max(0, x1)
            bx2 = min(w, bx1 + banner_w)

            overlay = annotated_img.copy()
            cv2.rectangle(overlay, (bx1, by1), (bx2, by2), box_color, -1)
            cv2.addWeighted(overlay, 0.85, annotated_img, 0.15, 0, annotated_img)

            cv2.putText(annotated_img, title, (bx1 + 6, by1 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_img, badge, (bx1 + 6, by1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (240, 240, 240), 1, cv2.LINE_AA)

        # 5. Save annotated image
        ann_filename = f"annotated_{image_id}.jpg"
        ann_path = os.path.join(CAPTURED_IMAGES_DIR, ann_filename)
        cv2.imwrite(ann_path, annotated_img)

        ann_img_res = firebase_service.save_image_to_firestore(
            image_input=annotated_img,
            image_id=f"ann_{image_id}",
            filename=ann_filename,
            metadata={"type": "uploaded_annotated", "filename": filename}
        )
        ann_url = ann_img_res.get("image_url", f"/captured_images/{ann_filename}")

        # 6. Save detection records into Firestore / Local Database
        timestamp_now = datetime.now().strftime("%d %b %Y, %I:%M:%S %p")
        raw_timestamp_now = datetime.now().isoformat()

        for ent in recognized_entities:
            rec_id = str(uuid.uuid4())
            cls_name = ent.get("class_name", "object")
            status = ent.get("status", "unknown").lower()
            is_person = (cls_name.lower() == "person")
            norm_status = "face_unavailable" if status in ["no_face", "face_unavailable"] else status

            # Extract dedicated clean crop for thumbnail
            cb = ent.get("face_bbox") if (is_person and status in ["known", "unknown"] and ent.get("face_bbox")) else ent.get("bbox", [0, 0, w, h])
            cx1, cy1, cx2, cy2 = cb
            cp_x = int((cx2 - cx1) * 0.12)
            cp_y = int((cy2 - cy1) * 0.12)
            tx1, ty1 = max(0, cx1 - cp_x), max(0, cy1 - cp_y)
            tx2, ty2 = min(w, cx2 + cp_x), min(h, cy2 + cp_y)
            thumb_crop = img[ty1:ty2, tx1:tx2]
            if thumb_crop.size == 0:
                thumb_crop = img

            th_filename = f"thumb_{rec_id[:8]}_{image_id}.jpg"
            th_path = os.path.join(CAPTURED_IMAGES_DIR, th_filename)
            cv2.imwrite(th_path, thumb_crop)
            th_res = firebase_service.save_image_to_firestore(
                image_input=thumb_crop,
                image_id=f"th_{rec_id[:8]}",
                filename=th_filename,
                metadata={"type": "upload_thumb", "event_id": rec_id}
            )
            th_url = th_res.get("image_url", f"/captured_images/{th_filename}")
            ent["thumbnail_url"] = th_url

            if is_person:
                p_id = ent.get("person_id") if status == "known" else ("UNKNOWN" if status == "unknown" else None)
                p_name = ent.get("name", "Unknown Visitor" if status == "unknown" else "Person (Face Unavailable)")
                dept = ent.get("department", "Visitor" if status == "unknown" else "General")
                role = ent.get("role", "Visitor" if status == "unknown" else "General")
                event_type = "known_person_entry" if status == "known" else ("unknown_security_alert" if status == "unknown" else "person_face_unavailable")
            else:
                p_id = None
                p_name = cls_name.capitalize()
                dept = "-"
                role = "Detected Object"
                event_type = f"object_detected_{cls_name}"

            ent["person_name"] = p_name

            det_record = {
                "id": rec_id,
                "camera_id": "UPLOAD-ANALYSIS",
                "camera_name": "Image Upload Analysis",
                "camera_location": "Dashboard Upload",
                "object_class": cls_name,
                "object_type": cls_name,
                "recognition_status": norm_status,
                "person_id": p_id,
                "person_name": p_name,
                "department": dept,
                "role": role,
                "yolo_confidence": ent.get("yolo_confidence", 0.0),
                "face_confidence": ent.get("face_confidence", 0.0),
                "similarity_score": ent.get("similarity_score", 0.0) if (is_person and status in ["known", "unknown"]) else 0.0,
                "confidence": ent.get("similarity_score" if (is_person and status in ["known", "unknown"]) else "yolo_confidence", 0.8),
                "decision_threshold": camera_manager.recognizer.similarity_threshold,
                "image_id": f"ann_{image_id}",
                "image_url": ann_url,
                "thumbnail_url": th_url,
                "local_image_path": ann_path,
                "local_thumb_path": th_path,
                "event_type": event_type,
                "timestamp": timestamp_now,
                "raw_timestamp": raw_timestamp_now
            }
            firebase_service.save_detection(det_record)

        return jsonify({
            "success": True,
            "message": f"Analysis complete. Identified {len(recognized_entities)} entities.",
            "image_id": image_id,
            "original_image_url": orig_url,
            "annotated_image_url": ann_url,
            "entities_count": len(recognized_entities),
            "results": recognized_entities,
            "timestamp": timestamp_now
        }), 200

    except Exception as e:
        logger.error("Image identification error: %s", e, exc_info=True)
        return jsonify({"success": False, "error": f"Image processing failed: {str(e)}"}), 500


@app.route("/api/settings/thresholds", methods=["GET", "POST"])
def api_thresholds():
    """Inspect and configure YOLO confidence and Face Recognition similarity thresholds."""
    if request.method == "POST":
        data = request.get_json(silent=True) or request.form.to_dict()
        if "yolo_confidence" in data:
            try:
                yolo_c = float(data["yolo_confidence"])
                camera_manager.yolo.set_confidence(yolo_c)
            except Exception:
                pass
        if "recognition_similarity" in data:
            try:
                rec_sim = float(data["recognition_similarity"])
                camera_manager.recognizer.similarity_threshold = max(0.5, min(0.98, rec_sim))
            except Exception:
                pass

    return jsonify({
        "success": True,
        "yolo_confidence": camera_manager.yolo.confidence_threshold,
        "recognition_similarity_threshold": camera_manager.recognizer.similarity_threshold,
        "decision_threshold": camera_manager.recognizer.similarity_threshold,
        "enrolled_gallery_count": len(camera_manager.recognizer.enrolled_gallery)
    }), 200


@app.route("/api/recognition/latest", methods=["GET"])
def api_latest_recognition():
    """Returns the latest person recognition result from the live camera."""
    latest = camera_manager.latest_recognition
    return jsonify({
        "success": True,
        "recognition": latest,
        "enrolled_count": len(camera_manager.recognizer.enrolled_gallery)
    }), 200


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    """Return recent alert notifications."""
    return jsonify({"success": True, "alerts": camera_manager.recent_alerts}), 200


# ---------------------------------------------------------
# Firebase Cloud Configuration API
# ---------------------------------------------------------

@app.route("/api/firebase/status", methods=["GET"])
def api_firebase_status():
    """Returns current Firebase Firestore and image storage status."""
    try:
        status = firebase_config.get_firebase_status()
        status["images_cached_count"] = len(firebase_service._firestore_images_cache)
        return jsonify({"success": True, "status": status}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/firebase/configure", methods=["POST"])
def api_firebase_configure():
    """Updates Firebase configuration using uploaded JSON file or text."""
    try:
        data = None
        if "file" in request.files:
            file = request.files["file"]
            if file and file.filename and file.filename.endswith(".json"):
                import json
                data = json.loads(file.read().decode("utf-8"))
        elif request.is_json:
            req_json = request.get_json()
            if "credentials_json" in req_json:
                import json
                data = json.loads(req_json["credentials_json"])
            elif "type" in req_json and req_json.get("type") == "service_account":
                data = req_json
        else:
            cred_text = request.form.get("credentials_json", "").strip()
            if cred_text:
                import json
                data = json.loads(cred_text)

        if not data:
            return jsonify({"success": False, "error": "No valid service account JSON provided"}), 400

        is_valid, msg = firebase_config.validate_credentials_dict(data)
        if not is_valid:
            return jsonify({"success": False, "error": f"Validation failed: {msg}"}), 400

        key_path = os.path.join(os.path.dirname(__file__), "serviceAccountKey.json")
        with open(key_path, "w", encoding="utf-8") as f:
            import json
            json.dump(data, f, indent=2)

        firebase_config.initialize_firebase(credentials_input=data, force_reinit=True)
        test_result = firebase_config.test_firestore_connection()

        return jsonify({
            "success": test_result.get("success", False),
            "message": test_result.get("message", "Firebase credentials saved!"),
            "error": test_result.get("error"),
            "status": firebase_config.get_firebase_status()
        }), (200 if test_result.get("success") else 400)

    except Exception as e:
        logger.error("Error configuring Firebase: %s", e)
        return jsonify({"success": False, "error": f"Failed to configure Firebase: {str(e)}"}), 500


@app.route("/api/firebase/test", methods=["POST"])
def api_firebase_test():
    """Tests the Cloud Firestore read/write connection."""
    res = firebase_config.test_firestore_connection()
    return jsonify(res), (200 if res.get("success") else 400)


# ---------------------------------------------------------
# Error Handlers
# ---------------------------------------------------------

@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Resource not found"}), 404
    return render_template("index.html", error="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal Server Error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal server error occurred"}), 500
    return render_template("index.html", error="An internal error occurred. Running in safe mode."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    host = os.environ.get("HOST", "0.0.0.0")

    import socket
    def is_port_in_use(p):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('127.0.0.1', p)) == 0

    if is_port_in_use(port):
        alt_port = port + 1
        if not is_port_in_use(alt_port):
            logger.warning("Port %d is occupied. Automatically using port %d.", port, alt_port)
            port = alt_port

    print(f"🚀 AI Camera Monitoring System running at http://127.0.0.1:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
