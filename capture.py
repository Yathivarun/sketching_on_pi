import cv2
import os
from datetime import datetime

# ---------------- Settings ----------------
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

print("\n=== IMAGE CAPTURE MODE ===")
print("Controls:")
print("  SPACE - Capture & Save Image")
print("  ESC   - Exit\n")

# ---------------- Get ID ----------------
person_id = input("Enter ID: ").strip()

if not person_id:
    print("ERROR: ID cannot be empty")
    exit(1)

# ---------------- Open Camera ----------------
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

print("\nCamera started...\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to read frame")
            break

        # Display instructions on frame
        cv2.putText(frame, "SPACE = Capture | ESC = Exit",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        cv2.imshow("Camera", frame)

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
