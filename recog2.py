import cv2
import numpy as np
import onnxruntime as ort
import os
import json
import time
import sys

# ---------------- Paths ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE = os.path.join(BASE, "glintr100.onnx")

# Database directory
DB_DIR = os.path.expanduser("~/face_db")

print("Detector :", SCRFD)
print("Recognizer:", ARCFACE)
print("Database  :", DB_DIR)

assert os.path.exists(SCRFD), "SCRFD model not found"
assert os.path.exists(ARCFACE), "GlintR100 model not found"

if not os.path.exists(DB_DIR):
    print(f"WARNING: Database directory not found: {DB_DIR}")
    print("Please run enrollment.py first to create face database")

# ---------------- ONNX Sessions ----------------
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

det_sess = ort.InferenceSession(SCRFD, opts, providers=["CPUExecutionProvider"])
rec_sess = ort.InferenceSession(ARCFACE, opts, providers=["CPUExecutionProvider"])

det_input = det_sess.get_inputs()[0].name
rec_input = rec_sess.get_inputs()[0].name

# ---------------- Load Database ----------------
def load_database():
    """Load all face embeddings from database"""
    database = {}
    
    if not os.path.exists(DB_DIR):
        return database
    
    for person_name in os.listdir(DB_DIR):
        person_dir = os.path.join(DB_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue
        
        embeddings = []
        for file in os.listdir(person_dir):
            if file.endswith('.npy'):
                emb_path = os.path.join(person_dir, file)
                emb = np.load(emb_path)
                embeddings.append(emb)
        
        if embeddings:
            # Average all embeddings for this person
            avg_embedding = np.mean(embeddings, axis=0)
            database[person_name] = {
                'embedding': avg_embedding,
                'count': len(embeddings)
            }
            print(f"Loaded {len(embeddings)} embedding(s) for: {person_name}")
    
    return database

# Load database
print("\n=== Loading Face Database ===")
face_database = load_database()
print(f"Total persons in database: {len(face_database)}\n")

# ---------------- Camera ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Could not open camera")
    exit(1)

# ---------------- Preprocess Functions ----------------
def preprocess_det(img, input_size=(640, 640)):
    img_resized = cv2.resize(img, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)
    return np.expand_dims(img_resized, axis=0)

def preprocess_rec(face):
    if face.size == 0:
        return None
    face = cv2.resize(face, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32)
    face = (face - 127.5) / 128.0
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face, axis=0)

def distance2bbox(points, distance, max_shape=None):
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
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.3):
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
    
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    
    if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
        return None
    
    return np.array([x1, y1, x2, y2]), float(best_score)

def detect_face(frame):
    """Detect the largest face in the frame"""
    try:
        det_input_data = preprocess_det(frame)
        det_out = det_sess.run(None, {det_input: det_input_data})
        result = scrfd_postprocess(det_out, frame.shape, thresh=0.3)
        return result
    except Exception as e:
        print(f"Detection error: {e}")
        return None

# ---------------- Face Recognition ----------------
def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings"""
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def recognize_face(embedding, database, threshold=0.3):
    """
    Recognize face by comparing with database
    Returns: (name, similarity) or (None, similarity) if no match
    
    Typical ArcFace thresholds:
    - 0.25-0.30: Strict (low false positives)
    - 0.30-0.40: Moderate
    - 0.40+: Relaxed (higher false positives)
    """
    if not database:
        return None, None
    
    best_match = None
    best_similarity = -1
    
    for name, data in database.items():
        db_embedding = data['embedding']
        similarity = cosine_similarity(embedding, db_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = name
    
    # Check if best match exceeds threshold
    if best_similarity >= threshold:
        return best_match, best_similarity
    else:
        return None, best_similarity

def process_capture(frame):
    """Process captured frame for face detection and recognition"""
    print("\n" + "="*60)
    print("📸 CAPTURING FRAME...")
    
    result = detect_face(frame)
    
    if result is None:
        print("❌ No face detected in captured frame")
        return None
    
    box, score = result
    x1, y1, x2, y2 = box
    
    face = frame[y1:y2, x1:x2]
    
    if face.size == 0 or face.shape[0] == 0 or face.shape[1] == 0:
        print("❌ Invalid face region")
        return None
    
    # Generate embedding
    face_input = preprocess_rec(face)
    if face_input is None:
        print("❌ Failed to preprocess face")
        return None
    
    emb = rec_sess.run(None, {rec_input: face_input})[0]
    
    # Recognize face
    name, similarity = recognize_face(emb, face_database, threshold=0.3)
    
    # Create result image with bounding box
    result_img = frame.copy()
    
    if name:
        # Known person
        color = (0, 255, 0)  # Green
        label = f"✓ {name}: {similarity:.3f}"
        status = "FOUND"
        print(f"✅ FACE FOUND: {name} (similarity: {similarity:.3f})")
    else:
        # Unknown person
        color = (0, 0, 255)  # Red
        sim_text = f"{similarity:.3f}" if similarity else "N/A"
        label = f"❌ Unknown: {sim_text}"
        status = "NOT FOUND"
        print(f"❌ FACE NOT FOUND in database (best similarity: {similarity:.3f})")
    
    # Draw bounding box
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 4)
    
    # Draw label with background
    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(result_img, (x1, y1 - label_h - 15), (x1 + label_w + 10, y1), color, -1)
    cv2.putText(result_img, label, (x1 + 5, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    print("="*60)
    
    return {
        'image': result_img,
        'box': box,
        'name': name,
        'similarity': similarity,
        'status': status
    }

# ---------------- Main Program ----------------
def main():
    print("\n" + "="*60)
    print("           FACE RECOGNITION - CAPTURE MODE")
    print("="*60)
    print("\nInstructions:")
    print("  • Press SPACE to capture current frame and recognize face")
    print("  • Press ESC to exit")
    print(f"\nDatabase contains {len(face_database)} person(s)")
    
    if len(face_database) == 0:
        print("\n⚠️  WARNING: No faces in database! Recognition will always show 'NOT FOUND'")
    
    capture_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Create display frame
        display_frame = frame.copy()
        
        # Show instructions on frame
        cv2.putText(display_frame, "SPACE: Capture & Recognize", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, "ESC: Exit", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"DB: {len(face_database)} persons", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("Face Recognition - Capture Mode", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("\nExiting...")
            break
        elif key == 32:  # SPACE
            capture_count += 1
            print(f"\n📸 Capture #{capture_count}")
            
            # Process the captured frame
            result = process_capture(frame)
            
            if result:
                # Show result window
                cv2.imshow(f"Recognition Result - Capture #{capture_count}", result['image'])
                
                # Wait for key press to continue
                print("\nPress any key to continue capturing...")
                cv2.waitKey(0)
                cv2.destroyWindow(f"Recognition Result - Capture #{capture_count}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print(f"Session ended. Total captures: {capture_count}")
    print("="*60)

if __name__ == "__main__":
    main()
