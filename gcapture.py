import cv2
import os
from datetime import datetime

# ---------------- Settings ----------------
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n=== RASPBERRY PI IMAGE CAPTURE ===")
print("Controls:")
print("  SPACE - Capture & Save Image")
print("  ESC   - Exit\n")

# ---------------- Get ID ----------------
person_id = input("Enter ID: ").strip()

if not person_id:
    print("ERROR: ID cannot be empty")
    exit(1)

# ---------------- GStreamer Pipeline ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! "
    "appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Could not open Raspberry Pi camera")
    exit(1)

print("\nCamera started...\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Display instructions
        cv2.putText(frame,
                    "SPACE = Capture | ESC = Exit",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2)

        cv2.imshow("Pi Camera", frame)

        key = cv2.waitKey(1)

        if key == 27:  # ESC
            break

        elif key == 32:  # SPACE
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{person_id}_{timestamp}.jpg"
            save_path = os.path.join(SAVE_DIR, filename)

            cv2.imwrite(save_path, frame)

            print(f"✓ Image saved: {save_path}")

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended")
