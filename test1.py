import cv2
import numpy as np
import onnxruntime as ort
import os
import json
import time
from datetime import datetime
from threading import Thread, Lock
import queue

# ================= CONFIGURATION =================
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE = os.path.join(BASE, "glintr100.onnx")

# Database directory
DB_DIR = os.path.expanduser("~/face_db")
os.makedirs(DB_DIR, exist_ok=True)

print("=" * 50)
print("FAST FACE ENROLLMENT SYSTEM")
print("=" * 50)
print(f"Detector : {SCRFD}")
print(f"Recognizer: {ARCFACE}")
print(f"Database  : {DB_DIR}")
print("=" * 50)

assert os.path.exists(SCRFD), f"SCRFD model not found at {SCRFD}"
assert os.path.exists(ARCFACE), f"GlintR100 model not found at {ARCFACE}"

# ================= OPTIMIZED ONNX SESSIONS =================
opts = ort.SessionOptions()
opts.intra_op_num_threads = 1  # Reduced threads
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

print("Loading face detection model...")
det_sess = ort.InferenceSession(SCRFD, opts, providers=["CPUExecutionProvider"])
print("Loading face recognition model...")
rec_sess = ort.InferenceSession(ARCFACE, opts, providers=["CPUExecutionProvider"])

det_input_name = det_sess.get_inputs()[0].name
rec_input_name = rec_sess.get_inputs()[0].name

print("✓ Models loaded successfully")

# ================= OPTIMIZED CAMERA SETUP =================
def setup_camera():
    """Setup optimized camera for Raspberry Pi"""
    # LOWER RESOLUTION for faster processing
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=640,height=480,framerate=30/1 ! "  # Reduced from 1640x1232
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("WARNING: Trying alternative pipeline...")
        pipeline = "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return None
    
    print(f"✓ Camera opened: 640x480 (optimized for speed)")
    return cap

# ================= OPTIMIZED DETECTION =================
def fast_face_detection(frame, det_sess):
    """Fast face detection with optimizations"""
    # Use smaller input for detection
    det_size = 320  # Reduced from 640
    h, w = frame.shape[:2]
    
    # Resize for detection (much faster)
    det_frame = cv2.resize(frame, (det_size, det_size))
    
    # Preprocess
    det_frame = det_frame.astype(np.float32)
    det_frame = (det_frame - 127.5) / 128.0
    det_frame = det_frame.transpose(2, 0, 1)
    det_frame = np.expand_dims(det_frame, axis=0)
    
    # Run detection
    try:
        outputs = det_sess.run(None, {det_input_name: det_frame})
        
        # Simple post-processing (extract first face only)
        # This assumes your model outputs in a certain format
        # Adjust based on your actual model output
        scores = outputs[0][0]  # Adjust indices based on your model
        bboxes = outputs[1][0]  # Adjust indices based on your model
        
        if len(scores) > 0:
            # Get highest confidence face
            max_idx = np.argmax(scores)
            if scores[max_idx] > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = bboxes[max_idx]
                
                # Scale back to original size
                x1 = int(x1 * w / det_size)
                y1 = int(y1 * h / det_size)
                x2 = int(x2 * w / det_size)
                y2 = int(y2 * h / det_size)
                
                # Ensure within bounds
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+10, min(x2, w))
                y2 = max(y1+10, min(y2, h))
                
                return (x1, y1, x2, y2), float(scores[max_idx])
    except Exception as e:
        print(f"Detection error: {e}")
    
    return None, 0.0

# ================= LIGHTWEIGHT FACE DETECTION (ALTERNATIVE) =================
def lightweight_face_detection(frame):
    """Even faster face detection using OpenCV's Haar Cascade"""
    # Convert to grayscale (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained face detector
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) > 0:
        # Get largest face
        x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
        return (x, y, x+w, y+h), 0.8  # Fixed confidence for cascade
    
    return None, 0.0

# ================= OPTIMIZED ENROLLMENT =================
def save_face_data(name, face_img, embedding):
    """Save face data efficiently"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save embedding
    embedding_file = os.path.join(person_dir, f"{timestamp}.npy")
    np.save(embedding_file, embedding)
    
    # Save face image (compressed)
    image_file = os.path.join(person_dir, f"{timestamp}.jpg")
    cv2.imwrite(image_file, face_img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    print(f"✓ Saved: {name}")
    return True

def generate_embedding(face_img, rec_sess):
    """Generate face embedding efficiently"""
    if face_img.size == 0:
        return None
    
    # Resize to model input size
    face_resized = cv2.resize(face_img, (112, 112))
    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_resized = face_resized.astype(np.float32)
    face_resized = (face_resized - 127.5) / 128.0
    face_resized = face_resized.transpose(2, 0, 1)
    face_resized = np.expand_dims(face_resized, axis=0)
    
    # Generate embedding
    embedding = rec_sess.run(None, {rec_input_name: face_resized})[0]
    return embedding

# ================= MAIN ENROLLMENT LOOP (OPTIMIZED) =================
def main():
    # Initialize camera
    print("\nInitializing camera...")
    cap = setup_camera()
    if cap is None:
        return
    
    print("\n" + "="*60)
    print("FAST FACE ENROLLMENT")
    print("="*60)
    print("INSTRUCTIONS:")
    print("1. Position face in camera view")
    print("2. Press 'c' to capture when ready")
    print("3. Enter name when prompted")
    print("4. Press 'q' to quit")
    print("="*60 + "\n")
    
    # State variables
    current_face = None
    face_box = None
    last_detection_time = 0
    detection_interval = 0.3  # Run detection every 300ms (not every frame)
    
    print("Starting... Press 'c' to capture, 'q' to quit")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            current_time = time.time()
            
            # Run detection at intervals (not every frame)
            if current_time - last_detection_time > detection_interval:
                # Use lightweight detection for preview
                face_box, confidence = lightweight_face_detection(frame)
                last_detection_time = current_time
                
                if face_box:
                    x1, y1, x2, y2 = face_box
                    current_face = frame[y1:y2, x1:x2].copy()
            
            # Draw minimal UI (no complex boxes)
            if face_box:
                x1, y1, x2, y2 = face_box
                
                # Simple rectangle (thin, no text for speed)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Simple instruction
                cv2.putText(display_frame, "READY - Press 'c'", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "NO FACE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show controls (static, not updated every frame)
            cv2.putText(display_frame, "'c'=Capture | 'q'=Quit", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display
            cv2.imshow("Face Enrollment", display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting...")
                break
                
            elif key == ord('c'):
                if current_face is not None and current_face.shape[0] > 30:
                    print("\n" + "="*50)
                    print("CAPTURING FACE...")
                    print("="*50)
                    
                    # Show captured face briefly
                    cv2.imshow("Captured Face", current_face)
                    cv2.waitKey(300)
                    cv2.destroyWindow("Captured Face")
                    
                    # Get name
                    name = input("Enter name (or Enter to skip): ").strip()
                    
                    if name:
                        # Generate embedding using your actual model
                        print("Generating embedding...")
                        embedding = generate_embedding(current_face, rec_sess)
                        
                        if embedding is not None:
                            # Save
                            save_face_data(name, current_face, embedding)
                            print(f"✓ Enrollment complete for: {name}\n")
                        else:
                            print("✗ Failed to generate embedding\n")
                    else:
                        print("✗ Skipped (no name)\n")
                    
                    # Reset for next capture
                    current_face = None
                    face_box = None
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nEnrollment session ended")

# ================= ULTRA-SIMPLE VERSION (MINIMAL UI) =================
def ultra_simple_enrollment():
    """Even simpler version - just show camera, press key to capture"""
    print("\n" + "="*60)
    print("ULTRA-SIMPLE FACE ENROLLMENT")
    print("="*60)
    print("Instructions:")
    print("1. Look at camera")
    print("2. Press SPACEBAR when ready")
    print("3. That's it!")
    print("="*60)
    
    # Simple camera setup
    pipeline = "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("Camera error!")
        return
    
    print("\nCamera ready. Press SPACE to capture, 'q' to quit\n")
    
    frame_count = 0
    last_print = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Calculate and display FPS every second
        current_time = time.time()
        if current_time - last_print >= 1.0:
            fps = frame_count / (current_time - last_print)
            print(f"FPS: {fps:.1f} - Press SPACE to capture", end='\r')
            frame_count = 0
            last_print = current_time
        
        # Minimal display - just the camera feed
        cv2.imshow("Press SPACE to Capture Face", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord(' '):  # SPACEBAR
            print("\n\n" + "="*50)
            print("CAPTURING...")
            
            # Simple face detection on captured frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                x, y, w, h = faces[0]  # Take first face
                face_img = frame[y:y+h, x:x+w]
                
                if face_img.shape[0] > 30:  # Valid face
                    # Show captured face
                    cv2.imshow("Captured Face", face_img)
                    cv2.waitKey(500)
                    
                    # Get name
                    name = input("\nEnter name for this face: ").strip()
                    
                    if name:
                        # Generate and save embedding
                        embedding = generate_embedding(face_img, rec_sess)
                        if embedding is not None:
                            save_face_data(name, face_img, embedding)
                            print(f"✓ Saved: {name}")
                        else:
                            print("✗ Failed to generate embedding")
                    else:
                        print("✗ Skipped")
                else:
                    print("✗ Face too small")
            else:
                print("✗ No face detected in capture")
            
            print("="*50 + "\n")
            print("Ready for next capture...")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")

# ================= CHOOSE MODE =================
if __name__ == "__main__":
    print("SELECT MODE:")
    print("1. Optimized Enrollment (with face detection preview)")
    print("2. Ultra-Simple (just camera, press space to capture)")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "2":
        ultra_simple_enrollment()
    else:
        main()
