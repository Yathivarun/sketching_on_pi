#!/usr/bin/env python3
"""
Camera Test Utility for Raspberry Pi
Use this to troubleshoot camera issues before running the main application.

Usage:
    python3 test_camera.py
    python3 test_camera.py --camera 1  # Try camera ID 1
"""

import cv2
import sys
import time
import argparse

def test_camera(camera_id=0, backend=None):
    """Test camera and display basic info."""
    
    print("="*60)
    print("🎥 Camera Test Utility")
    print("="*60)
    print(f"Testing camera ID: {camera_id}")
    print("")
    
    # Step 1: Check /dev/video* devices
    print("[1/5] Checking for video devices...")
    try:
        import os
        video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
        if video_devices:
            print(f"✓ Found devices: {', '.join(['/dev/'+d for d in video_devices])}")
        else:
            print("✗ No video devices found in /dev/")
            print("  Make sure camera is connected")
            return False
    except:
        print("⚠️ Cannot check /dev/ directory")
    
    print("")
    
    # Step 2: Try to open camera
    print(f"[2/5] Opening camera {camera_id}...")
    
    if backend:
        print(f"     Using backend: {backend}")
        cap = cv2.VideoCapture(camera_id, backend)
    else:
        print(f"     Using auto backend")
        cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_id}")
        print("\nTroubleshooting:")
        print("  • Try different camera ID: python3 test_camera.py --camera 1")
        print("  • For Pi Camera Module:")
        print("    - Run: sudo raspi-config")
        print("    - Select: Interface Options → Camera → Enable")
        print("    - Reboot")
        print("  • Check permissions: sudo usermod -a -G video $USER")
        print("  • Test with: raspistill -o test.jpg (for Pi Camera)")
        return False
    
    print("✓ Camera opened successfully")
    print("")
    
    # Step 3: Get camera properties
    print("[3/5] Camera properties:")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    backend_name = cap.getBackendName()
    
    print(f"  Backend: {backend_name}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print("")
    
    # Step 4: Try to set resolution
    print("[4/5] Testing resolution change...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if new_width == 640 and new_height == 480:
        print(f"✓ Resolution set to 640x480")
    else:
        print(f"⚠️ Requested 640x480, got {new_width}x{new_height}")
        print("  This is OK for some cameras")
    print("")
    
    # Step 5: Test frame capture
    print("[5/5] Testing frame capture...")
    
    # Warmup
    for i in range(10):
        cap.read()
        time.sleep(0.05)
    
    # Capture test frames
    successful = 0
    failed = 0
    
    for i in range(30):
        ret, frame = cap.read()
        if ret and frame is not None:
            successful += 1
        else:
            failed += 1
        time.sleep(0.033)  # ~30 FPS
    
    print(f"  Captured: {successful}/30 frames")
    print(f"  Failed: {failed}/30 frames")
    
    if successful == 0:
        print("✗ Cannot capture any frames")
        cap.release()
        return False
    elif failed > 10:
        print("⚠️ Many failed frames - camera may be unstable")
    else:
        print("✓ Frame capture working")
    
    print("")
    
    # Display test
    print("[DISPLAY] Opening preview window...")
    print("  Press 'q' to quit, 's' to save test image")
    print("")
    
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Test", 640, 480)
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("⚠️ Failed to read frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        elapsed = time.time() - start_time
        
        if elapsed > 0:
            actual_fps = frame_count / elapsed
        else:
            actual_fps = 0
        
        # Add overlay
        display = frame.copy()
        cv2.putText(
            display,
            f"Camera {camera_id} - {frame.shape[1]}x{frame.shape[0]} @ {actual_fps:.1f} FPS",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        cv2.putText(
            display,
            "Press 'q' to quit, 's' to save",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        cv2.imshow("Camera Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n[TEST] Quit by user")
            break
        elif key == ord('s'):
            filename = f"test_camera_{camera_id}.jpg"
            cv2.imwrite(filename, frame)
            print(f"\n[TEST] ✓ Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("")
    print("="*60)
    print("✓ CAMERA TEST COMPLETE")
    print("="*60)
    print("")
    print("Summary:")
    print(f"  Camera ID: {camera_id}")
    print(f"  Backend: {backend_name}")
    print(f"  Resolution: {new_width}x{new_height}")
    print(f"  Capture Success Rate: {successful}/{successful+failed}")
    print(f"  Average FPS: {actual_fps:.1f}")
    print("")
    
    if successful > 20:
        print("✅ Camera is working well - ready for main application")
        return True
    else:
        print("⚠️ Camera is working but may be unstable")
        return True


def main():
    parser = argparse.ArgumentParser(description='Test camera on Raspberry Pi')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--backend', type=str, default=None, 
                       choices=['auto', 'v4l2', 'gstreamer'],
                       help='Camera backend (default: auto)')
    
    args = parser.parse_args()
    
    # Map backend string to OpenCV constant
    backend_map = {
        'auto': None,
        'v4l2': cv2.CAP_V4L2,
        'gstreamer': cv2.CAP_GSTREAMER,
    }
    
    backend = backend_map.get(args.backend)
    
    try:
        success = test_camera(args.camera, backend)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n[TEST] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
