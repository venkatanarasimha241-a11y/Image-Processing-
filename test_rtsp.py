"""
test_rtsp.py
------------
Simple standalone RTSP camera test utility for OpenCV.
Verifies your CCTV / IP Camera RTSP stream connectivity, resolution, and FPS
before integrating into the complete AI monitoring pipeline.

Usage:
    python test_rtsp.py
    python test_rtsp.py --url "rtsp://admin:password@192.168.1.100:554/stream1"
    python test_rtsp.py --headless  # Test stream without opening GUI window
"""

import os
import sys
import time
import argparse
import cv2

# Try to load credentials from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Tell pytest not to collect this standalone script as a test module
__test__ = False


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone RTSP Stream Tester")
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("RTSP_URL", ""),
        help="RTSP Stream URL (e.g. rtsp://user:pass@192.168.1.100:554/stream)"
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="tcp",
        choices=["tcp", "udp"],
        help="RTSP Transport protocol (default: tcp)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless terminal mode without opening GUI window"
    )
    return parser.parse_args()


def test_rtsp_stream(rtsp_url: str, transport: str = "tcp", headless: bool = False):
    if not rtsp_url:
        print("\n" + "=" * 60)
        print("  📹 RTSP / WEBCAM TEST PROGRAM")
        print("=" * 60)
        print("Enter your CCTV RTSP URL or camera index:")
        print("  - For RTSP: rtsp://user:pass@192.168.1.100:554/stream1")
        print("  - For Local Webcam: 0 (or press Enter to test webcam 0)")
        try:
            rtsp_url = input("\nEnter RTSP URL or Webcam Index [0]: ").strip()
        except (EOFError, KeyboardInterrupt):
            rtsp_url = ""
        
        if not rtsp_url:
            rtsp_url = "0"
            print("Using default local webcam index 0.")

    # Check if target is a local device index (e.g. "0")
    is_device_index = False
    device_idx = 0
    try:
        device_idx = int(rtsp_url)
        is_device_index = True
    except (ValueError, TypeError):
        is_device_index = False

    # Sanitize URL for console display (hide password)
    masked_url = rtsp_url
    if not is_device_index and "@" in rtsp_url:
        prefix, host_part = rtsp_url.split("@", 1)
        if "://" in prefix:
            proto, creds = prefix.split("://", 1)
            user = creds.split(":")[0] if ":" in creds else creds
            masked_url = f"{proto}://{user}:******@{host_part}"

    print("\n" + "=" * 60)
    print("  📹 CAMERA STREAM TEST PROGRAM")
    print("=" * 60)
    print(f"Target Source   : {masked_url} {'(Webcam Index ' + str(device_idx) + ')' if is_device_index else '(RTSP Network Stream)'}")
    print(f"Transport       : {transport.upper()}")
    print(f"Backend Driver  : {'Default OS Camera' if is_device_index else 'cv2.CAP_FFMPEG'}")
    print(f"Display Mode    : {'Headless (CLI Only)' if headless else 'GUI Window (cv2.imshow)'}")
    print("-" * 60)

    # Configure OpenCV FFmpeg capture options for RTSP streams
    if not is_device_index:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            f"rtsp_transport;{transport}|fflags;nobuffer|flags;low_delay|max_delay;500000"
        )

    print("Connecting to camera... (timeout: 10s)")
    start_connect = time.time()

    try:
        if is_device_index:
            cap = cv2.VideoCapture(device_idx)
        else:
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    except Exception as e:
        print(f"❌ Failed to invoke cv2.VideoCapture: {e}")
        return False

    connect_time = time.time() - start_connect

    if not cap.isOpened():
        print(f"\n❌ FAILED TO OPEN RTSP STREAM (took {connect_time:.2f}s)")
        print("\nTroubleshooting Checklist:")
        print("  1. Ping camera IP: Is the camera reachable from this computer?")
        print("  2. Verify RTSP port (usually 554) is open and not blocked by firewall.")
        print("  3. Double-check username & password credentials in URL.")
        print("  4. Check stream path (e.g. /h264Preview_01_main, /live/ch0, /Streaming/Channels/101).")
        print("  5. Verify camera supports RTSP over TCP / H.264 video codec.")
        print("  6. Test the URL in VLC Media Player: Media -> Open Network Stream.")
        return False

    print(f"✅ Connection established in {connect_time:.2f}s!")

    # Read first test frame
    ret, frame = cap.read()
    if not ret or frame is None:
        print("❌ Connected, but failed to read initial frame from camera.")
        cap.release()
        return False

    h, w = frame.shape[:2]
    fps_prop = cap.get(cv2.CAP_PROP_FPS) or 0.0
    print(f"✅ First frame received successfully!")
    print(f"   - Resolution: {w} x {h}")
    print(f"   - Stream Reported FPS: {fps_prop:.1f}")
    print("-" * 60)

    if headless:
        print("Running in headless verification mode (reading 50 frames)...")
        frame_count = 0
        t0 = time.time()
        while frame_count < 50:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"⚠️ Warning: Frame read returned None at frame #{frame_count}")
                time.sleep(0.05)
                continue
            frame_count += 1
            if frame_count % 10 == 0:
                elapsed = time.time() - t0
                real_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"   Frame #{frame_count:03d} | Resolution: {frame.shape[1]}x{frame.shape[0]} | Real-time FPS: {real_fps:.1f}")

        cap.release()
        print("\n🎉 RTSP CCTV STREAM VERIFIED SUCCESSFULLY IN HEADLESS MODE!")
        return True

    # GUI display mode
    window_name = f"RTSP Stream Test - {masked_url}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(w, 960), min(h, 540))

    print("📺 Displaying live stream in window. Press 'q' or ESC in video window to exit.")

    frame_count = 0
    fps = 0.0
    last_fps_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Frame drop or camera disconnected. Retrying...")
                time.sleep(0.1)
                continue

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # Overlay telemetry on test video
            overlay = frame.copy()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            info_text = f"RTSP LIVE | {w}x{h} | {fps:.1f} FPS | {timestamp}"
            cv2.rectangle(overlay, (10, 10), (450, 45), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, info_text, (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                print("\nUser requested exit.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. RTSP test completed.")

    return True


if __name__ == "__main__":
    args = parse_args()
    test_rtsp_stream(rtsp_url=args.url, transport=args.transport, headless=args.headless)
