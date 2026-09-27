"""
firebase_config.py
------------------
Handles Firebase Admin SDK initialization for Cloud Firestore and Firestore Image Storage.
Gracefully degrades to DEMO MODE if credentials are missing or placeholders.
Supports dynamic re-initialization and live connection testing.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FirebaseConfig")

# Global Firebase references
_firebase_app = None
_db = None
_bucket = None
_is_demo_mode = False
_project_id = ""
_client_email = ""

SERVICE_ACCOUNT_FILE = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
STORAGE_BUCKET_NAME = os.environ.get("FIREBASE_STORAGE_BUCKET", "")
PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")


def validate_credentials_dict(data: dict) -> Tuple[bool, str]:
    """Validate whether dictionary contains legitimate Firebase service account keys."""
    if not isinstance(data, dict):
        return False, "Credentials must be a JSON object"

    if data.get("type") != "service_account":
        return False, "Key 'type' must be 'service_account'"

    project_id = data.get("project_id", "")
    if not project_id or "placeholder" in project_id.lower() or "demo" in project_id.lower() and "placeholder" in str(data):
        return False, "Invalid or placeholder project_id"

    private_key = data.get("private_key", "")
    if not private_key or "PLACEHOLDER" in private_key or "BEGIN PRIVATE KEY" not in private_key:
        return False, "Invalid or placeholder private_key"

    client_email = data.get("client_email", "")
    if not client_email or "@" not in client_email:
        return False, "Invalid client_email in credentials"

    return True, "Valid service account credentials"


def is_valid_credential_file(filepath: str) -> bool:
    """Check if the service account key exists and is not a placeholder template."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            is_valid, _ = validate_credentials_dict(data)
            return is_valid
    except Exception as e:
        logger.warning("Error reading credential file %s: %s", filepath, e)
        return False


def initialize_firebase(credentials_input: Optional[Any] = None, force_reinit: bool = False):
    """
    Initializes Firebase Admin SDK for Firestore.
    Accepts optional dictionary, JSON string, or file path.
    Activates Demo Mode if credentials are absent or invalid.
    """
    global _firebase_app, _db, _bucket, _is_demo_mode, _project_id, _client_email

    if _firebase_app is not None and not force_reinit:
        return _is_demo_mode

    # Delete existing app if forcing re-initialization
    if force_reinit and _firebase_app is not None:
        try:
            import firebase_admin
            firebase_admin.delete_app(_firebase_app)
        except Exception as e:
            logger.warning("Could not delete previous Firebase app: %s", e)
        _firebase_app = None
        _db = None
        _bucket = None

    # Check if firebase_admin is installed
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
    except ImportError:
        logger.warning("firebase-admin package is not installed. Entering DEMO MODE.")
        _is_demo_mode = True
        return _is_demo_mode

    cred_obj = None
    extracted_project_id = ""
    extracted_email = ""

    # 1. Custom credentials passed directly
    if credentials_input is not None:
        if isinstance(credentials_input, str):
            try:
                # Try parsing as JSON string first
                cred_dict = json.loads(credentials_input)
                is_valid, msg = validate_credentials_dict(cred_dict)
                if is_valid:
                    cred_obj = credentials.Certificate(cred_dict)
                    extracted_project_id = cred_dict.get("project_id", "")
                    extracted_email = cred_dict.get("client_email", "")
                else:
                    logger.warning("Passed JSON string failed validation: %s", msg)
            except Exception:
                # Might be a file path
                if is_valid_credential_file(credentials_input):
                    cred_obj = credentials.Certificate(credentials_input)
        elif isinstance(credentials_input, dict):
            is_valid, msg = validate_credentials_dict(credentials_input)
            if is_valid:
                cred_obj = credentials.Certificate(credentials_input)
                extracted_project_id = credentials_input.get("project_id", "")
                extracted_email = credentials_input.get("client_email", "")
            else:
                logger.warning("Passed dict failed validation: %s", msg)

    # 2. Check environment variable for raw JSON
    if cred_obj is None:
        env_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if env_json:
            try:
                cred_dict = json.loads(env_json)
                is_valid, _ = validate_credentials_dict(cred_dict)
                if is_valid:
                    cred_obj = credentials.Certificate(cred_dict)
                    extracted_project_id = cred_dict.get("project_id", "")
                    extracted_email = cred_dict.get("client_email", "")
            except Exception as e:
                logger.warning("Failed to parse FIREBASE_SERVICE_ACCOUNT_JSON: %s", e)

    # 3. Check local serviceAccountKey.json file
    if cred_obj is None and is_valid_credential_file(SERVICE_ACCOUNT_FILE):
        try:
            with open(SERVICE_ACCOUNT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                extracted_project_id = data.get("project_id", "")
                extracted_email = data.get("client_email", "")
            cred_obj = credentials.Certificate(SERVICE_ACCOUNT_FILE)
        except Exception as e:
            logger.warning("Failed reading %s: %s", SERVICE_ACCOUNT_FILE, e)

    # Initialize SDK if valid credentials obtained
    if cred_obj is not None:
        try:
            options = {}
            if STORAGE_BUCKET_NAME:
                options["storageBucket"] = STORAGE_BUCKET_NAME

            _firebase_app = firebase_admin.initialize_app(cred_obj, options)
            _db = firestore.client()
            _project_id = extracted_project_id or PROJECT_ID
            _client_email = extracted_email

            if STORAGE_BUCKET_NAME:
                try:
                    _bucket = storage.bucket()
                except Exception:
                    _bucket = None
            else:
                _bucket = None

            _is_demo_mode = False
            logger.info("Firebase initialized successfully in LIVE Cloud mode (Project: %s).", _project_id)
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin SDK: %s. Falling back to DEMO MODE.", e)
            _is_demo_mode = True
    else:
        logger.info("Valid service account credentials not found. Operating in DEMO MODE.")
        _is_demo_mode = True

    return _is_demo_mode


def get_firestore_client():
    """Return Firestore client if initialized, else None."""
    global _db
    if _db is None and not _is_demo_mode:
        initialize_firebase()
    return _db


def get_storage_bucket():
    """Return Firebase Storage bucket if initialized, else None."""
    global _bucket
    if _bucket is None and not _is_demo_mode:
        initialize_firebase()
    return _bucket


def is_demo_mode() -> bool:
    """Return True if system is running in demo/offline mode."""
    global _is_demo_mode
    if _firebase_app is None and not _is_demo_mode:
        initialize_firebase()
    return _is_demo_mode


def get_firebase_status() -> Dict[str, Any]:
    """Returns detailed status of Firebase connection and Firestore image storage."""
    global _is_demo_mode, _db, _project_id, _client_email
    if _firebase_app is None and not _is_demo_mode:
        initialize_firebase()

    connected = (not _is_demo_mode) and (_db is not None)
    return {
        "connected": connected,
        "is_demo_mode": _is_demo_mode,
        "mode": "LIVE Cloud Firestore" if connected else "DEMO MODE (Local Fallback)",
        "project_id": _project_id or ("Demo Project" if _is_demo_mode else "Not Configured"),
        "client_email": _client_email or "offline@local.demo",
        "database": "Cloud Firestore",
        "image_storage": "Cloud Firestore (Base64 JPEG in 'images' collection)",
        "service_account_file_exists": os.path.exists(SERVICE_ACCOUNT_FILE)
    }


def test_firestore_connection() -> Dict[str, Any]:
    """Tests live Firestore read and write operations."""
    db = get_firestore_client()
    if db is None or is_demo_mode():
        return {
            "success": False,
            "error": "Firebase is running in Demo Mode. Service account credentials required.",
            "mode": "DEMO MODE"
        }

    try:
        from datetime import datetime
        test_ref = db.collection("_healthcheck").document("ping")
        test_ref.set({
            "ping": True,
            "timestamp": datetime.now().isoformat(),
            "test_message": "Firestore connectivity verified successfully"
        })
        doc = test_ref.get()
        if doc.exists:
            test_ref.delete()
            return {
                "success": True,
                "message": "Successfully connected and verified read/write to Cloud Firestore!",
                "project_id": _project_id
            }
        return {
            "success": False,
            "error": "Document written to Firestore could not be verified."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Firestore verification failed: {str(e)}"
        }

