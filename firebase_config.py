"""
firebase_config.py
------------------
Handles Firebase Admin SDK initialization for Firestore and Cloud Storage.
Gracefully degrades to DEMO MODE if credentials are missing or placeholders.
"""

import os
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FirebaseConfig")

# Global Firebase references
_firebase_app = None
_db = None
_bucket = None
_is_demo_mode = False

SERVICE_ACCOUNT_FILE = os.environ.get("FIREBASE_SERVICE_ACCOUNT_PATH", "serviceAccountKey.json")
STORAGE_BUCKET_NAME = os.environ.get("FIREBASE_STORAGE_BUCKET", "")
PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "")


def is_valid_credential_file(filepath: str) -> bool:
    """Check if the service account key exists and is not a placeholder template."""
    if not os.path.exists(filepath):
        return False
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Check for dummy placeholder values
            private_key = data.get("private_key", "")
            if not private_key or "PLACEHOLDER" in private_key:
                return False
            if not data.get("project_id") or "placeholder" in data.get("project_id", "").lower():
                return False
            return True
    except Exception as e:
        logger.warning("Error reading credential file %s: %s", filepath, e)
        return False


def initialize_firebase():
    """Initializes Firebase Admin SDK or activates Demo Mode if credentials are absent."""
    global _firebase_app, _db, _bucket, _is_demo_mode

    if _firebase_app is not None or _is_demo_mode:
        return _is_demo_mode

    # Check if firebase_admin is installed
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore, storage
    except ImportError:
        logger.warning("firebase-admin package is not installed. Entering DEMO MODE.")
        _is_demo_mode = True
        return _is_demo_mode

    # Check for credentials
    if is_valid_credential_file(SERVICE_ACCOUNT_FILE):
        try:
            cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
            options = {}
            if STORAGE_BUCKET_NAME:
                options["storageBucket"] = STORAGE_BUCKET_NAME

            _firebase_app = firebase_admin.initialize_app(cred, options)
            _db = firestore.client()
            if STORAGE_BUCKET_NAME:
                _bucket = storage.bucket()
            else:
                try:
                    _bucket = storage.bucket()
                except Exception:
                    _bucket = None
                    logger.info("Storage bucket not configured. Evidence images will be stored locally.")

            _is_demo_mode = False
            logger.info("Firebase initialized successfully in LIVE Cloud mode.")
        except Exception as e:
            logger.warning("Failed to initialize Firebase Admin SDK: %s. Falling back to DEMO MODE.", e)
            _is_demo_mode = True
    else:
        logger.info("Valid serviceAccountKey.json not found. Operating in DEMO MODE.")
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
