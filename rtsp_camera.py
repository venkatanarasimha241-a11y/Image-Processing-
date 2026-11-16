"""
rtsp_camera.py
--------------
Production-grade RTSP / IP Camera and Video Stream Engine for OpenCV.

Features:
1. Flexible connection: RTSP stream URL, IP camera HTTP/MJPEG, or local USB webcam index.
2. Connection verification: verify_connection() tests stream connectivity before committing.
3. Non-blocking asynchronous frame grabber thread to prevent OpenCV RTSP socket buffer lag.
4. Automatic stream loss detection with exponential backoff auto-reconnect.
5. Thread-safe operations with dedicated thread synchronization.
6. Multi-camera support: Multiple RTSPCamera instances indexed by camera_id.
7. Safe credential masking: Passwords in RTSP URLs are never exposed in logs or API payloads.
8. Clean resource lifecycle: start(), stop(), disconnect(), and release().
"""

import os
import cv2
import time
import logging
import threading
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("RTSPCamera")


def mask_rtsp_url(url: Optional[str]) -> str:
    """Masks username:password in RTSP URLs for secure logging and UI display."""
    if not url:
        return "None (Webcam / Synthetic)"
    url_str = url.strip()
    if "@" in url_str:
        prefix, host_part = url_str.split("@", 1)
        if "://" in prefix:
            proto, creds = prefix.split("://", 1)
            user = creds.split(":")[0] if ":" in creds else creds
            return f"{proto}://{user}:******@{host_part}"
    return url_str


class RTSPCamera:
    """
    Manages an individual RTSP CCTV / IP Camera or Webcam video capture stream.
    Runs a high-priority background frame grabber to pull frames and eliminate buffer delay.
    """
    def __init__(
        self,
        camera_id: str = "gate_camera_1",
        source: Optional[str] = None,
        name: str = "CCTV Camera",
        location: str = "Main Entrance",
        transport: str = "tcp"
    ):
        self.camera_id = camera_id
        self.source = source or ""  # RTSP URL or integer device index as string
        self.name = name
        self.location = location
        self.transport = transport.lower()

        # Operational state
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.connection_status = "INITIALIZING"  # CONNECTED, CONNECTING, RECONNECTING, OFFLINE, DISCONNECTED
        self.last_error = ""
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 30
        self.last_frame_time = 0.0

        # Frame buffers
        self._current_frame: Optional[Any] = None
        self._frame_lock = threading.Lock()
        self._cap_lock = threading.Lock()

        # Threads
        self._grabber_thread: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None

        # Metrics
        self.fps = 0.0
        self._frame_count = 0
        self._last_fps_calc = time.time()
        self.resolution: Tuple[int, int] = (0, 0)
        self.is_webcam = False

    @staticmethod
    def verify_stream(source: Optional[str], transport: str = "tcp", timeout_seconds: float = 6.0) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Tests whether an RTSP URL or camera source is reachable and decodable without altering active instances.
        Returns: (success: bool, message: str, stream_info: dict)
        """
        if not source or not source.strip():
            return False, "No camera source provided", {}

        clean_source = source.strip()
        is_device_index = False
        device_idx = 0
        try:
            device_idx = int(clean_source)
            is_device_index = True
        except (ValueError, TypeError):
            is_device_index = False

        if not is_device_index:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"
            )

        start_time = time.time()
        test_cap = None
        try:
            if is_device_index:
                test_cap = cv2.VideoCapture(device_idx)
            else:
                test_cap = cv2.VideoCapture(clean_source, cv2.CAP_FFMPEG)
                if not test_cap.isOpened():
                    test_cap = cv2.VideoCapture(clean_source)

            if not test_cap.isOpened():
                elapsed = time.time() - start_time
                return False, f"Failed to open camera stream (timed out after {elapsed:.1f}s). Check IP, port 554, and credentials.", {}

            ret, frame = test_cap.read()
            if not ret or frame is None or frame.size == 0:
                return False, "Connected to stream, but failed to decode video frames.", {}

            h, w = frame.shape[:2]
            fps = test_cap.get(cv2.CAP_PROP_FPS) or 25.0
            return True, "Camera verified successfully!", {
                "width": w,
                "height": h,
                "fps": round(fps, 1),
                "is_device_index": is_device_index
            }

        except Exception as e:
            return False, f"Camera verification error: {str(e)}", {}
        finally:
            if test_cap is not None:
                try:
                    test_cap.release()
                except Exception:
                    pass

    def start(self) -> bool:
        """Starts video capture and background grabber thread."""
        if self.is_running and self._grabber_thread and self._grabber_thread.is_alive():
            logger.info("Camera [%s] already running.", self.camera_id)
            return True

        self.is_running = True
        success = self._open_source()

        # Start grabber thread
        self._grabber_thread = threading.Thread(
            target=self._grabber_loop,
            daemon=True,
            name=f"Grabber-{self.camera_id}"
        )
        self._grabber_thread.start()
        logger.info("Camera [%s] pipeline started (Status: %s)", self.camera_id, self.connection_status)
        return success

    def stop(self):
        """Stops the camera stream and grabber thread, keeping configuration."""
        self.is_running = False
        if self._grabber_thread and self._grabber_thread.is_alive() and threading.current_thread() != self._grabber_thread:
            try:
                self._grabber_thread.join(timeout=1.0)
            except Exception:
                pass

        self._release_capture()
        self.connection_status = "OFFLINE"
        logger.info("Camera [%s] stopped.", self.camera_id)

    def disconnect(self):
        """Completely disconnects and marks as DISCONNECTED."""
        self.stop()
        self.connection_status = "DISCONNECTED"
        with self._frame_lock:
            self._current_frame = None
        logger.info("Camera [%s] disconnected.", self.camera_id)

    def _open_source(self) -> bool:
        """Opens cv2.VideoCapture with optimized FFmpeg flags."""
        self._release_capture()

        if not self.source or not self.source.strip():
            # Fallback to local webcam
            self.source = "0"

        clean_source = self.source.strip()
        try:
            device_idx = int(clean_source)
            self.is_webcam = True
            source_target = device_idx
        except (ValueError, TypeError):
            self.is_webcam = False
            source_target = clean_source

        self.connection_status = "CONNECTING"
        logger.info("Opening camera [%s] -> %s", self.camera_id, mask_rtsp_url(self.source))

        if not self.is_webcam:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;{self.transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"
            )

        with self._cap_lock:
            try:
                if self.is_webcam:
                    self.cap = cv2.VideoCapture(source_target)
                    if self.cap and self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                else:
                    self.cap = cv2.VideoCapture(source_target, cv2.CAP_FFMPEG)
                    if not self.cap or not self.cap.isOpened():
                        self.cap = cv2.VideoCapture(source_target)

                if self.cap and self.cap.isOpened():
                    try:
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except Exception:
                        pass

                    ret, frame = self.cap.read()
                    if ret and frame is not None and frame.size > 0:
                        self.connection_status = "CONNECTED"
                        self.reconnect_attempts = 0
                        self.last_error = ""
                        self.resolution = (frame.shape[1], frame.shape[0])
                        with self._frame_lock:
                            self._current_frame = frame
                        logger.info("Camera [%s] opened successfully! Res: %dx%d", self.camera_id, self.resolution[0], self.resolution[1])
                        return True
                    else:
                        self.last_error = "Camera opened but failed to read initial frame"
                        self.connection_status = "RECONNECTING"
                        return False
                else:
                    self.last_error = "Could not open video capture device"
                    self.connection_status = "RECONNECTING"
                    return False
            except Exception as e:
                self.last_error = f"Exception opening camera: {str(e)}"
                self.connection_status = "RECONNECTING"
                logger.error("Error opening camera [%s]: %s", self.camera_id, e)
                return False

    def _release_capture(self):
        """Safely releases cv2.VideoCapture."""
        with self._cap_lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception as e:
                    logger.debug("Error releasing VideoCapture: %s", e)
                self.cap = None

    def _grabber_loop(self):
        """Continuously pulls latest frames to keep socket buffer latency near zero."""
        read_failures = 0

        while self.is_running:
            ret, frame = False, None
            with self._cap_lock:
                if self.cap is not None and self.cap.isOpened():
                    try:
                        ret, frame = self.cap.read()
                    except Exception as e:
                        ret, frame = False, None
                        self.last_error = str(e)

            if ret and frame is not None and frame.size > 0:
                read_failures = 0
                now = time.time()
                self.last_frame_time = now
                self.connection_status = "CONNECTED"

                with self._frame_lock:
                    self._current_frame = frame
                    self.resolution = (frame.shape[1], frame.shape[0])

                # Calculate real FPS
                self._frame_count += 1
                elapsed = now - self._last_fps_calc
                if elapsed >= 1.0:
                    self.fps = round(self._frame_count / elapsed, 1)
                    self._frame_count = 0
                    self._last_fps_calc = now

                # Yield small amount of time to allow other threads to run
                time.sleep(0.005)
            else:
                read_failures += 1
                time.sleep(0.05)

                if read_failures >= 15:
                    # Connection dropped
                    self.connection_status = "RECONNECTING"
                    logger.warning("Camera [%s] stream lost. Initiating auto-reconnect...", self.camera_id)
                    self._trigger_reconnect()
                    read_failures = 0

    def _trigger_reconnect(self):
        """Runs auto-reconnection in a separate daemon thread to avoid blocking."""
        if self._reconnect_thread and self._reconnect_thread.is_alive():
            return

        def _reconnect_worker():
            backoff = 2.0
            while self.is_running and self.connection_status == "RECONNECTING":
                self.reconnect_attempts += 1
                logger.info("Camera [%s] reconnect attempt #%d (backoff %.1fs)...",
                            self.camera_id, self.reconnect_attempts, backoff)
                if self._open_source():
                    logger.info("Camera [%s] reconnected successfully!", self.camera_id)
                    break
                time.sleep(backoff)
                backoff = min(15.0, backoff * 1.5)

        self._reconnect_thread = threading.Thread(
            target=_reconnect_worker,
            daemon=True,
            name=f"Reconnect-{self.camera_id}"
        )
        self._reconnect_thread.start()

    def read_frame(self) -> Tuple[bool, Optional[Any]]:
        """Returns the freshest frame read by the grabber thread."""
        with self._frame_lock:
            if self._current_frame is not None and self._current_frame.size > 0:
                return True, self._current_frame.copy()
        return False, None

    def update_source(self, new_source: str, new_name: Optional[str] = None, new_location: Optional[str] = None):
        """Updates RTSP URL or camera source and re-establishes connection."""
        self.source = new_source
        if new_name:
            self.name = new_name
        if new_location:
            self.location = new_location
        if self.is_running:
            self._open_source()

    def get_status(self) -> Dict[str, Any]:
        """Returns camera telemetry and connection information."""
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "location": self.location,
            "masked_url": mask_rtsp_url(self.source),
            "connection_status": self.connection_status,
            "is_running": self.is_running,
            "is_webcam": self.is_webcam,
            "fps": self.fps,
            "resolution": f"{self.resolution[0]}x{self.resolution[1]}" if self.resolution[0] > 0 else "N/A",
            "last_error": self.last_error,
            "reconnect_attempts": self.reconnect_attempts
        }
