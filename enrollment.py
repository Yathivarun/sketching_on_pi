import cv2
import numpy as np
import onnxruntime as ort
import os
import json
import time
from datetime import datetime

# ================= CONFIGURATION =================
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE = os.path.join(BASE, "glintr100.onnx")

# Database directory
DB_DIR = os.path.expanduser("~/face_db")
os.makedirs(DB_DIR, exist_ok=True)

print("=" * 50)
print("FACE ENROLLMENT SYSTEM")
print("=" * 50)
print(f"Detector : {SCRFD}")
print(f"Recognizer: {ARCFACE}")
print(f"Database  : {DB_DIR}")
print("=" * 50)

assert os.path.exists(SCRFD), f"SCRFD model not found at {SCRFD}"
assert os.path.exists(ARCFACE), f"GlintR100 model not found at {ARCFACE}"

# ================= ONNX SESSIONS =================
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

print("Loading face detection model...")
det_sess = ort.InferenceSession(SCRFD, opts, providers=["CPUExecutionProvider"])
print("Loading face recognition model...")
rec_sess = ort.InferenceSession(ARCFACE, opts, providers=["CPUExecutionProvider"])

det_input = det_sess.get_inputs()[0].name
rec_input = rec_sess.get_inputs()[0].name

print("✓ Models loaded successfully")

# ================= CAMERA SETUP =================
def setup_camera():
    """Setup Raspberry Pi camera with GStreamer"""
    # Your working pipeline
    pipeline = (
        "libcamerasrc ! "
        "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("WARNING: GStreamer pipeline failed, trying default...")
        # Fallback to default
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: Could not open any camera!")
            return None
    
    print(f"✓ Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    return cap

# ================= PREPROCESS FUNCTIONS =================
def preprocess_det(img, input_size=(640, 640)):
    """Preprocess image for face detection"""
    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)
    return np.expand_dims(img_resized, axis=0)

def preprocess_rec(face_img):
    """Preprocess face image for recognition"""
    if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
        return None
    
    # Resize to 112x112 as required by ArcFace
    face_resized = cv2.resize(face_img, (112, 112))
    face_resized = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_resized = face_resized.astype(np.float32)
    face_resized = (face_resized - 127.5) / 128.0
    face_resized = face_resized.transpose(2, 0, 1)
    return np.expand_dims(face_resized, axis=0)

# ================= FACE DETECTION POST-PROCESSING =================
def distance2bbox(points, distance, max_shape=None):
    """Convert distance to bounding box"""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    
    return np.stack([x1, y1, x2, y2], axis=-1)

def generate_anchors(height, width, stride, num_anchors=2):
    """Generate anchor points for face detection"""
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.5):
    """Post-process SCRFD model output"""
    h, w = orig_shape[:2]
    fmc = 3
    feat_stride_fpn = [8, 16, 32]
    num_anchors = 2
    
    all_boxes = []
    all_scores = []
    outputs_per_scale = len(outputs) // fmc
    
    for idx in range(fmc):
        stride = feat_stride_fpn[idx]
        fm_height = input_size // stride
        fm_width = input_size // stride
        
        score_idx = idx * outputs_per_scale
        bbox_idx = score_idx + 1
        
        if score_idx >= len(outputs) or bbox_idx >= len(outputs):
            continue
        
        scores = outputs[score_idx]
        bboxes = outputs[bbox_idx]
        
        if len(scores.shape) == 3:
            scores = scores[0]
        if len(bboxes.shape) == 3:
            bboxes = bboxes[0]
        
        scores_flat = scores.flatten()
        
        if len(bboxes.shape) == 1:
            num_boxes = len(bboxes) // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        elif len(bboxes.shape) == 2 and bboxes.shape[1] == 4:
            bboxes_reshaped = bboxes
        else:
            total_elements = bboxes.size
            num_boxes = total_elements // 4
            bboxes_reshaped = bboxes.reshape(num_boxes, 4)
        
        num_valid = min(len(scores_flat), len(bboxes_reshaped))
        if num_valid == 0:
            continue
        
        scores_matched = scores_flat[:num_valid]
        bboxes_matched = bboxes_reshaped[:num_valid]
        
        total_positions = fm_height * fm_width * num_anchors
        
        if num_valid != total_positions:
            actual_positions = num_valid // num_anchors
            if actual_positions * num_anchors < num_valid:
                actual_positions += 1
            actual_fm_size = int(np.sqrt(actual_positions / num_anchors))
            if actual_fm_size == 0:
                actual_fm_size = 1
            
            anchors_temp = []
            count = 0
            for y in range(actual_fm_size):
                for x in range(actual_fm_size):
                    for _ in range(num_anchors):
                        if count >= num_valid:
                            break
                        anchors_temp.append([x * stride + stride // 2, y * stride + stride // 2])
                        count += 1
                    if count >= num_valid:
                        break
                if count >= num_valid:
                    break
            
            while len(anchors_temp) < num_valid:
                anchors_temp.append(anchors_temp[-1] if anchors_temp else [stride // 2, stride // 2])
            
            anchor_centers = np.array(anchors_temp[:num_valid], dtype=np.float32)
        else:
            anchor_centers = generate_anchors(fm_height, fm_width, stride, num_anchors)
            anchor_centers = anchor_centers[:num_valid]
        
        if len(anchor_centers) != len(scores_matched):
            min_len = min(len(anchor_centers), len(scores_matched), len(bboxes_matched))
            anchor_centers = anchor_centers[:min_len]
            scores_matched = scores_matched[:min_len]
            bboxes_matched = bboxes_matched[:min_len]
        
        try:
            valid_mask = scores_matched > thresh
            if not np.any(valid_mask):
                continue
            
            valid_scores = scores_matched[valid_mask]
            valid_bboxes = bboxes_matched[valid_mask]
            valid_anchors = anchor_centers[valid_mask]
            
            decoded_boxes = distance2bbox(valid_anchors, valid_bboxes)
            all_scores.extend(valid_scores)
            all_boxes.extend(decoded_boxes)
        except Exception as e:
            continue
    
    if len(all_boxes) == 0:
        return None
    
    all_scores = np.array(all_scores)
    all_boxes = np.array(all_boxes)
    
    # Get the highest confidence face
    best_idx = np.argmax(all_scores)
    best_box = all_boxes[best_idx]
    best_score = all_scores[best_idx]
    
    scale_x = w / input_size
    scale_y = h / input_size
    
    x1, y1, x2, y2 = best_box
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    # Ensure box is within image bounds
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 20:
        return None
    
    return np.array([x1, y1, x2, y2]), float(best_score)

# ================= ENROLLMENT FUNCTIONS =================
def save_embedding(name, embedding, face_image, timestamp):
    """Save face embedding and metadata to database"""
    person_dir = os.path.join(DB_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save embedding
    embedding_file = os.path.join(person_dir, f"{timestamp}.npy")
    np.save(embedding_file, embedding)
    
    # Save face image
    image_file = os.path.join(person_dir, f"{timestamp}.jpg")
    cv2.imwrite(image_file, face_image)
    
    # Save metadata
    metadata = {
        "name": name,
        "timestamp": timestamp,
        "embedding_shape": embedding.shape,
        "embedding_file": embedding_file,
        "image_file": image_file
    }
    
    metadata_file = os.path.join(person_dir, f"{timestamp}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata

def enroll_person(name, face_image, embedding):
    """Enroll a new person with given face and embedding"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"ENROLLING: {name}")
    print(f"{'='*60}")
    
    metadata = save_embedding(name, embedding, face_image, timestamp)
    
    print(f"✓ Saved embedding: {metadata['embedding_file']}")
    print(f"✓ Saved face image: {metadata['image_file']}")
    print(f"✓ Metadata saved: {metadata['metadata_file']}")
    print(f"{'='*60}")
    
    return True

# ================= MAIN ENROLLMENT LOOP =================
def main():
    # Initialize camera
    print("\nInitializing camera...")
    cap = setup_camera()
    if cap is None:
        return
    
    print("\n" + "="*60)
    print("FACE ENROLLMENT SYSTEM READY")
    print("="*60)
    print("INSTRUCTIONS:")
    print("1. Position face in the camera view")
    print("2. Face will be auto-detected (green box)")
    print("3. Press 'CAPTURE' button to enroll detected face")
    print("4. Press 's' to save current frame as enrollment")
    print("5. Press 'q' to quit")
    print("="*60 + "\n")
    
    # State variables
    current_face = None
    current_box = None
    current_embedding = None
    capture_mode = False
    enrollment_count = 0
    
    # Create window
    cv2.namedWindow("Face Enrollment", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Enrollment", 800, 600)
    
    # Create button position (bottom center)
    button_text = "CAPTURE"
    button_width = 200
    button_height = 50
    
    print("Starting face detection...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                time.sleep(0.1)
                continue
            
            display_frame = frame.copy()
            h, w = display_frame.shape[:2]
            
            # Run face detection
            face_detected = False
            try:
                det_input_data = preprocess_det(frame)
                det_out = det_sess.run(None, {det_input: det_input_data})
                result = scrfd_postprocess(det_out, frame.shape, thresh=0.5)
                
                if result is not None:
                    box, score = result
                    x1, y1, x2, y2 = box
                    
                    # Extract face region
                    face = frame[y1:y2, x1:x2]
                    
                    if face.size > 0 and face.shape[0] > 20 and face.shape[1] > 20:
                        face_detected = True
                        
                        # Generate embedding for this face
                        face_input = preprocess_rec(face)
                        if face_input is not None:
                            emb = rec_sess.run(None, {rec_input: face_input})[0]
                            
                            # Store current face data
                            current_face = face.copy()
                            current_box = box.copy()
                            current_embedding = emb.copy()
                            
                            # Draw bounding box (green for detected)
                            color = (0, 255, 0)  # Green
                            thickness = 3
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            # Draw face info
                            info_text = f"Face: {score:.2f}"
                            cv2.putText(display_frame, info_text, (x1, y1 - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # Draw capture hint
                            cv2.putText(display_frame, "Press 's' to enroll or click CAPTURE button", 
                                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                            
            except Exception as e:
                print(f"Detection error: {e}")
            
            # Draw capture button (always visible)
            button_x = w // 2 - button_width // 2
            button_y = h - button_height - 20
            
            # Button background
            button_color = (0, 150, 255) if face_detected else (100, 100, 100)
            cv2.rectangle(display_frame, 
                         (button_x, button_y),
                         (button_x + button_width, button_y + button_height),
                         button_color, -1)
            
            # Button border
            cv2.rectangle(display_frame,
                         (button_x, button_y),
                         (button_x + button_width, button_y + button_height),
                         (255, 255, 255), 2)
            
            # Button text
            text_size = cv2.getTextSize(button_text, cv2.FONT_HERSHEY_DUPLEX, 1, 2)[0]
            text_x = button_x + (button_width - text_size[0]) // 2
            text_y = button_y + (button_height + text_size[1]) // 2
            cv2.putText(display_frame, button_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
            
            # Draw instructions
            cv2.putText(display_frame, f"Enrolled: {enrollment_count} faces", 
                       (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow("Face Enrollment", display_frame)
            
            # Handle mouse events for button click
            def mouse_callback(event, x, y, flags, param):
                nonlocal current_face, current_embedding, enrollment_count, capture_mode
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    if (button_x <= x <= button_x + button_width and 
                        button_y <= y <= button_y + button_height):
                        if face_detected and current_face is not None and current_embedding is not None:
                            capture_mode = True
            
            cv2.setMouseCallback("Face Enrollment", mouse_callback)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting enrollment system...")
                break
                
            elif key == ord('s') or capture_mode:
                # Manual capture triggered
                if face_detected and current_face is not None and current_embedding is not None:
                    print("\n" + "="*50)
                    print("FACE CAPTURED!")
                    print("="*50)
                    
                    # Show captured face in separate window
                    cv2.imshow("Captured Face", current_face)
                    cv2.waitKey(500)  # Show for 500ms
                    cv2.destroyWindow("Captured Face")
                    
                    # Get name for enrollment
                    print("\nEnter name for this person (or press Enter to skip):")
                    name = input("Name: ").strip()
                    
                    if name:
                        # Enroll the person
                        if enroll_person(name, current_face, current_embedding):
                            enrollment_count += 1
                            print(f"\n✓ Enrollment successful! Total: {enrollment_count}")
                            
                            # Reset for next capture
                            current_face = None
                            current_embedding = None
                            capture_mode = False
                            
                            # Short pause to show success
                            time.sleep(1)
                    else:
                        print("✗ Enrollment skipped (no name provided)")
                        capture_mode = False
                else:
                    print("✗ No face detected! Please position face properly.")
                    capture_mode = False
            
            elif key == ord('r'):
                # Reset current face
                current_face = None
                current_embedding = None
                print("Face data reset")
    
    except KeyboardInterrupt:
        print("\n\nEnrollment interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*60)
        print("ENROLLMENT SESSION SUMMARY")
        print("="*60)
        print(f"Total faces enrolled: {enrollment_count}")
        print(f"Database location: {DB_DIR}")
        print("="*60)
        
        # Show database contents
        if os.path.exists(DB_DIR):
            print("\nDatabase contents:")
            for person in os.listdir(DB_DIR):
                person_path = os.path.join(DB_DIR, person)
                if os.path.isdir(person_path):
                    embeddings = [f for f in os.listdir(person_path) if f.endswith('.npy')]
                    print(f"  {person}: {len(embeddings)} embeddings")

if __name__ == "__main__":
    main()
