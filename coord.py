import cv2
import time

def quick_camera_test():
    """Quick test to verify camera is working"""
    print("Testing Raspberry Pi Camera...")
    
    # Your working pipeline
    pipeline = "libcamerasrc ! video/x-raw,width=1640,height=1232,framerate=30/1 ! videoconvert ! appsink"
    
    print(f"Using pipeline: {pipeline}")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("ERROR: Failed to open camera!")
        print("\nTrying alternative resolutions...")
        
        # Try different resolutions
        resolutions = [
            (640, 480),
            (800, 600),
            (1280, 720),
            (1920, 1080)
        ]
        
        for width, height in resolutions:
            pipeline = f"libcamerasrc ! video/x-raw,width={width},height={height} ! videoconvert ! appsink"
            print(f"Trying {width}x{height}...")
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                print(f"SUCCESS with {width}x{height}!")
                break
            time.sleep(0.5)
    
    if cap.isOpened():
        print("\nCamera is working!")
        print(f"Frame width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"Frame height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        
        # Capture a few frames to verify
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                print(f"Frame {i+1}: {frame.shape}")
                cv2.imshow(f'Test Frame {i+1}', frame)
                cv2.waitKey(300)
            else:
                print(f"Failed to capture frame {i+1}")
        
        cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Camera test passed! You can run the adjustment tool.")
        return True
    else:
        print("\n✗ Camera test failed!")
        print("\nTroubleshooting steps:")
        print("1. Check camera connection")
        print("2. Enable camera: sudo raspi-config -> Interface Options -> Camera")
        print("3. Reboot after enabling camera")
        print("4. Install GStreamer: sudo apt install libgstreamer1.0-0 gstreamer1.0-plugins-base gstreamer1.0-plugins-good")
        return False

if __name__ == "__main__":
    quick_camera_test()
