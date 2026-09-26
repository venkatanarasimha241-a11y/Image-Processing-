"""
app.py
------
AI Camera Monitoring System - Main Flask Application Server.
Provides web dashboard templates, live video streaming, and REST APIs
for persons management, detection logs, analytics, and camera controls.
"""

import os
import logging
from datetime import datetime
import cv2
import numpy as np
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
from camera import CameraManager, generate_mjpeg_stream
from recognition import KNOWN_FACES_DIR

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ai-camera-monitoring-secret-key-2026")

# Ensure required directories exist
CAPTURED_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "captured_images")
KNOWN_FACES_DIR = os.path.join(os.path.dirname(__file__), "known_faces")
os.makedirs(CAPTURED_IMAGES_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Initialize Firebase SDK or Demo Fallback
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
        # If any matching, display department scoped; otherwise display all with indicator
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
def video_feed():
    """Live multipart MJPEG video stream with YOLO bounding boxes."""
    return Response(
        generate_mjpeg_stream(),
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
        email = data.get("email", "").strip()
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
    """Returns real-time KPI statistics and chart metrics."""
    try:
        stats = firebase_service.get_system_stats()
        cam_status = camera_manager.get_status()
        stats["camera_fps"] = cam_status.get("fps", 0.0)
        stats["camera_is_synthetic"] = cam_status.get("is_synthetic", False)
        return jsonify({"success": True, "stats": stats}), 200
    except Exception as e:
        logger.error("Stats API error: %s", e)
        return jsonify({"success": False, "error": "Failed to fetch stats"}), 500


@app.route("/api/detections", methods=["GET"])
def api_detections():
    """Returns detection records with optional query parameters."""
    try:
        limit = int(request.args.get("limit", 50))
        filters = {
            "status": request.args.get("status"),
            "camera": request.args.get("camera"),
            "search": request.args.get("search")
        }
        # Clean null values
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
        data = {}
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        name = data.get("name", "").strip()
        department = data.get("department", "").strip()
        role = data.get("role", "Student").strip()
        person_id = data.get("person_id", "").strip()
        roll_number = data.get("roll_number", "").strip()
        class_name = data.get("class", "").strip()
        year = data.get("year", "").strip()
        cgpa = data.get("cgpa", "").strip()
        faculty_id = data.get("faculty_id", "").strip()
        designation = data.get("designation", "").strip()

        if not name:
            return jsonify({"success": False, "error": "Person name is required"}), 400

        # Handle face photo upload if provided
        image_url = "/static/images/default_avatar.svg"
        if "photo" in request.files:
            file = request.files["photo"]
            if file and file.filename != "":
                filename = f"{person_id or 'STU'}_{name.replace(' ', '_')}_{file.filename}"
                save_path = os.path.join(KNOWN_FACES_DIR, filename)
                file.save(save_path)
                image_url = f"/known_faces/{filename}"
                # Refresh face recognizer database
                camera_manager.recognizer.load_known_faces()

        new_person = {
            "person_id": person_id,
            "name": name,
            "department": department or "Computer Science & Engineering",
            "role": role,
            "image_url": image_url,
            "status": "ACTIVE"
        }

        if "student" in role.lower():
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
            return jsonify({"success": success, "message": f"Person {person_id} deleted"}), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

    # PUT: Update
    try:
        updates = request.get_json(silent=True) or request.form.to_dict()
        updated = firebase_service.update_person(person_id, updates)
        if updated:
            return jsonify({"success": True, "person": updated}), 200
        return jsonify({"success": False, "error": "Person not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/identify_image", methods=["POST"])
def api_identify_image():
    """Receives an uploaded photo and identifies person returning timetable & details."""
    try:
        if "image" not in request.files:
            return jsonify({"success": False, "error": "No image file provided"}), 400

        file = request.files["image"]
        if not file or file.filename == "":
            return jsonify({"success": False, "error": "Empty image file"}), 400

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "error": "Invalid image content"}), 400

        result = camera_manager.identify_image_array(img)
        return jsonify({"success": True, "data": result}), 200
    except Exception as e:
        logger.error("Image identification error: %s", e)
        return jsonify({"success": False, "error": "Identification pipeline failed"}), 500


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


@app.route("/api/capture", methods=["POST"])
def api_capture():
    """Capture instant manual snapshot."""
    snapshot = camera_manager.capture_snapshot()
    if snapshot:
        return jsonify({"success": True, "message": "Snapshot captured", "snapshot": snapshot}), 200
    return jsonify({"success": False, "error": "Camera frame not available for capture"}), 400


@app.route("/api/alerts", methods=["GET"])
def api_alerts():
    """Return recent alert notifications."""
    alerts = camera_manager.recent_alerts
    return jsonify({"success": True, "alerts": alerts}), 200


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
    print(f"🚀 AI Camera Monitoring System running at http://127.0.0.1:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)
