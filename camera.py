"""
camera.py
---------
Backward-compatible wrapper module forwarding to camera_manager.CameraManager
and rtsp_camera.RTSPCamera.
"""

from camera_manager import CameraManager, generate_mjpeg_stream, TrackedEntity
from rtsp_camera import RTSPCamera, mask_rtsp_url

__all__ = ["CameraManager", "generate_mjpeg_stream", "TrackedEntity", "RTSPCamera", "mask_rtsp_url"]
