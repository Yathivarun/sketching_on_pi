"""
Face Detection System with Network Communication
Raspberry Pi - Sensor Node with Integrated Display

Detects faces, generates embeddings, sends to laptop for matching.
Receives and displays person's images when match found.
Shows stock slideshow when idle.

Phase 2B Features:
- Integrated pi_display.py for showing images
- Dual window mode (sensor preview + display)
- Automatic slideshow/POI switching

FIXED: Image channel mismatch (BGRA to BGR conversion)
"""

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from datetime import datetime
import json
import glob
import threading

# Import network protocol and display
from network_protocol import PiClient
from pi_display import PiDisplay

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================
CONFIG = {
    "camera": {
        "width": 1640,
        "height": 1232,
        "framerate": 30,
        "capture_interval": 4.0  # 4 seconds between detections (as required)
    },
    "detection": {
        "input_size": (640, 640),
        "threshold": 0.3
    },
    "recognition": {
        "face_size": (112, 112),
    },
    "paths": {
        "models": os.path.expanduser("~/.insightface/models/light"),
        "stock_images": "stock_images"  # Directory for slideshow images
    },
    "network": {
        "laptop_ip": "192.168.137.1",
        "port": 5000,
        "enabled": True,  # Set to False to disable network (local mode)
        "auto_reconnect": True,
        "reconnect_delay": 5  # seconds
    },
    "display": {
        "show_preview": True,  # Show camera feed + detections
        "preview_window": "Pi Sensor - Face Detection",
        "enable_display": True,  # Enable display window for images
        "fullscreen": False  # Set True for HDMI fullscreen, False for windowed test
    }
}

# ============================================================================
# MODEL PATHS
# ============================================================================
SCRFD_MODEL = os.path.join(CONFIG["paths"]["models"], "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(CONFIG["paths"]["models"], "glintr100.onnx")

# ============================================================================
# GLOBAL STATE
# ============================================================================
class SensorState:
    def __init__(self):
        self.network_client = None
        self.connected = False
        self.last_detection_time = 0
        self.detection_count = 0
        self.match_count = 0
        self.current_display_images = []  # Images to show from laptop
        self.display_mode = "idle"  # "idle" or "person"
        self.last_person_id = None
        self.display_window = None  # PiDisplay instance

state = SensorState()

# ============================================================================
# INITIALIZATION
# ============================================================================
def initialize_models():
    """Initialize ONNX models for detection and recognition"""
    print("=" * 60)
    print("INITIALIZING FACE DETECTION MODELS")
    print("=" * 60)
    
    # Check models exist
    assert os.path.exists(SCRFD_MODEL), f"SCRFD model not found: {SCRFD_MODEL}"
    assert os.path.exists(ARCFACE_MODEL), f"ArcFace model not found: {ARCFACE_MODEL}"
    print(f"✓ Models found")
    print(f"  Detection: {SCRFD_MODEL}")
    print(f"  Recognition: {ARCFACE_MODEL}")
    
    # Initialize ONNX sessions
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 2
    
    det_sess = ort.InferenceSession(
        SCRFD_MODEL, 
        opts, 
        providers=["CPUExecutionProvider"]
    )
    
    rec_sess = ort.InferenceSession(
        ARCFACE_MODEL, 
        opts, 
        providers=["CPUExecutionProvider"]
    )
    
    # Get input names and shapes
    det_input_name = det_sess.get_inputs()[0].name
    det_input_shape = det_sess.get_inputs()[0].shape
    rec_input_name = rec_sess.get_inputs()[0].name
    rec_input_shape = rec_sess.get_inputs()[0].shape
    
    print(f"✓ ONNX sessions initialized")
    print(f"  Detection input: {det_input_name} {det_input_shape}")
    print(f"  Recognition input: {rec_input_name} {rec_input_shape}")
    
    return det_sess, rec_sess, det_input_name, rec_input_name

def initialize_display():
    """Initialize Pi display window"""
    if not CONFIG["display"]["enable_display"]:
        print("[DISPLAY] Display disabled in config")
        return None
    
    print("\n" + "=" * 60)
    print("INITIALIZING DISPLAY WINDOW")
    print("=" * 60)
    
    display = PiDisplay(
        fullscreen=CONFIG["display"]["fullscreen"],
        stock_dir=CONFIG["paths"]["stock_images"]
    )
    
    # Start display thread
    display.start()
    
    state.display_window = display
    print("✓ Display window started")
    
    return display

def initialize_network():
    """Initialize network client to connect to laptop"""
    if not CONFIG["network"]["enabled"]:
        print("[NETWORK] Network disabled in config")
        return None
    
    print("\n" + "=" * 60)
    print("INITIALIZING NETWORK CONNECTION")
    print("=" * 60)
    
    client = PiClient(
        laptop_ip=CONFIG["network"]["laptop_ip"],
        port=CONFIG["network"]["port"]
    )
    
    # Set callbacks
    client.on_match_result = handle_match_result
    client.on_images_received = handle_images_received
    client.on_disconnected = handle_disconnection
    
    # Try to connect
    if client.connect(timeout=10):
        state.network_client = client
        state.connected = True
        print("✓ Connected to laptop!")
        return client
    else:
        print("✗ Failed to connect to laptop")
        if CONFIG["network"]["auto_reconnect"]:
            print(f"  Will retry in {CONFIG['network']['reconnect_delay']}s...")
        return None

# ============================================================================
# NETWORK CALLBACKS
# ============================================================================
def handle_match_result(msg):
    """Called when laptop sends match result"""
    hit = msg.get("hit")
    person_id = msg.get("person_id")
    score = msg.get("score")
    
    if hit:
        state.match_count += 1
        state.last_person_id = person_id
        print(f"\n✅ MATCH! Person #{person_id} (score: {score:.3f})")
        print("   Waiting for images from laptop...")
    else:
        print(f"\n❌ No match (best score: {score:.3f})")

def handle_images_received(image_list):
    """Called when laptop sends images to display"""
    if not image_list:
        # Empty list = return to slideshow/idle
        state.display_mode = "idle"
        state.current_display_images = []
        print("← Laptop: Return to slideshow")
        
        # Update display window
        if state.display_window:
            state.display_window.return_to_slideshow()
    else:
        state.display_mode = "person"
        state.current_display_images = image_list
        print(f"← Received {len(image_list)} images from laptop")
        print(f"   Now displaying Person #{state.last_person_id}")
        
        # Update display window
        if state.display_window:
            state.display_window.show_poi_images(image_list)

def handle_disconnection():
    """Called when connection to laptop is lost"""
    state.connected = False
    state.network_client = None
    print("\n✗ Lost connection to laptop!")
    
    # Return display to slideshow
    if state.display_window:
        state.display_window.return_to_slideshow()
    
    if CONFIG["network"]["auto_reconnect"]:
        # Start reconnection in background thread
        threading.Thread(target=attempt_reconnect, daemon=True).start()

def attempt_reconnect():
    """Background thread to reconnect to laptop"""
    while CONFIG["network"]["auto_reconnect"] and not state.connected:
        print(f"[NETWORK] Reconnecting in {CONFIG['network']['reconnect_delay']}s...")
        time.sleep(CONFIG["network"]["reconnect_delay"])
        
        client = initialize_network()
        if client:
            print("[NETWORK] ✓ Reconnected successfully!")
            break

# ============================================================================
# CAMERA INITIALIZATION
# ============================================================================
def initialize_camera():
    """Initialize Pi Camera"""
    print("\n" + "=" * 60)
    print("INITIALIZING CAMERA")
    print("=" * 60)
    
    try:
        from picamera2 import Picamera2
        
        picam = Picamera2()
        
        # Configure camera
        config = picam.create_preview_configuration(
            main={"size": (CONFIG["camera"]["width"], CONFIG["camera"]["height"])},
            controls={"FrameRate": CONFIG["camera"]["framerate"]}
        )
        picam.configure(config)
        
        # Start camera
        picam.start()
        time.sleep(1)  # Warmup
        
        print(f"✓ Camera initialized ({CONFIG['camera']['width']}x{CONFIG['camera']['height']})")
        return picam
        
    except ImportError:
        print("✗ picamera2 not found, using fallback (webcam)")
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["camera"]["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["camera"]["height"])
        time.sleep(1)
        return cap

# ============================================================================
# IMAGE PREPROCESSING UTILITIES
# ============================================================================
def ensure_bgr_format(image):
    """
    Ensure image is in BGR format (3 channels)
    Handles BGRA (4 channels) and grayscale (1 channel) conversions
    
    CRITICAL FIX: PiCamera2 sometimes outputs BGRA format which causes
    ONNX model errors. This function normalizes to BGR.
    """
    if len(image.shape) == 2:
        # Grayscale to BGR
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 4:
            # BGRA to BGR (remove alpha channel)
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] == 3:
            # Already BGR
            return image
        else:
            raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

# ============================================================================
# DETECTION & RECOGNITION
# ============================================================================
def preprocess_detection(image):
    """
    Preprocess image for face detection (SCRFD format)
    
    FIXED: Added channel normalization to handle BGRA/grayscale inputs
    """
    # Ensure 3-channel BGR format
    image = ensure_bgr_format(image)
    
    input_size = CONFIG["detection"]["input_size"]
    img_resized = cv2.resize(image, input_size)
    img_resized = img_resized.astype(np.float32)
    img_resized = (img_resized - 127.5) / 128.0
    img_resized = img_resized.transpose(2, 0, 1)  # HWC to CHW
    
    blob = np.expand_dims(img_resized, axis=0)
    
    # Verify shape before returning
    expected_shape = (1, 3, input_size[0], input_size[1])
    if blob.shape != expected_shape:
        raise ValueError(f"Preprocessed blob shape {blob.shape} != expected {expected_shape}")
    
    return blob

def distance2bbox(points, distance, max_shape=None):
    """Convert distance predictions to bounding boxes"""
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
    """Generate anchor points for detection"""
    anchors = []
    for y in range(height):
        for x in range(width):
            for _ in range(num_anchors):
                anchors.append([x * stride + stride // 2, y * stride + stride // 2])
    return np.array(anchors, dtype=np.float32)

def scrfd_postprocess(outputs, orig_shape, input_size=640, thresh=0.3):
    """Postprocess SCRFD detection outputs"""
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
        
        anchors = generate_anchors(fm_height, fm_width, stride, num_anchors)
        if len(anchors) > len(bboxes_matched):
            anchors = anchors[:len(bboxes_matched)]
        elif len(anchors) < len(bboxes_matched):
            bboxes_matched = bboxes_matched[:len(anchors)]
            scores_matched = scores_matched[:len(anchors)]
        
        pred_boxes = distance2bbox(anchors, bboxes_matched * stride)
        
        scale_x = w / input_size
        scale_y = h / input_size
        pred_boxes[:, [0, 2]] *= scale_x
        pred_boxes[:, [1, 3]] *= scale_y
        
        valid_idx = scores_matched > thresh
        all_boxes.append(pred_boxes[valid_idx])
        all_scores.append(scores_matched[valid_idx])
    
    if not all_boxes:
        return []
    
    boxes = np.vstack(all_boxes)
    scores = np.concatenate(all_scores)
    
    # NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(),
        scores.tolist(),
        thresh,
        0.4
    )
    
    faces = []
    if len(indices) > 0:
        for i in indices.flatten():
            faces.append({
                "bbox": boxes[i].astype(int).tolist(),
                "score": float(scores[i])
            })
    
    return faces

def detect_faces(det_sess, det_input_name, image):
    """Run face detection"""
    input_size = CONFIG["detection"]["input_size"][0]
    
    try:
        blob = preprocess_detection(image)
        outputs = det_sess.run(None, {det_input_name: blob})
        faces = scrfd_postprocess(outputs, image.shape, input_size, CONFIG["detection"]["threshold"])
        return faces
    except Exception as e:
        print(f"ERROR in detect_faces: {e}")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")
        raise

def extract_face_aligned(image, bbox):
    """Extract and align face for recognition"""
    # Ensure BGR format
    image = ensure_bgr_format(image)
    
    x1, y1, x2, y2 = bbox
    
    # Add margin
    margin = 20
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(image.shape[1], x2 + margin)
    y2 = min(image.shape[0], y2 + margin)
    
    # Crop face
    face_img = image[y1:y2, x1:x2]
    
    # Resize to model input size
    face_size = CONFIG["recognition"]["face_size"]
    face_resized = cv2.resize(face_img, face_size)
    
    return face_resized

def preprocess_face(face_img):
    """
    Prepare face for ArcFace model
    
    FIXED: Corrected preprocessing normalization
    """
    if face_img.size == 0:
        return None
    
    # Ensure BGR format first
    face_img = ensure_bgr_format(face_img)
    
    # Resize
    face = cv2.resize(face_img, CONFIG["recognition"]["face_size"])
    
    # Convert BGR to RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    face = face.astype(np.float32) / 255.0
    
    # Apply ArcFace normalization: (x - 0.5) / 0.5 = 2x - 1
    face = (face - 0.5) / 0.5  # This transforms to [-1, 1] range
    
    # Transpose to CHW format
    face = face.transpose(2, 0, 1)  # HWC to CHW
    
    # Add batch dimension
    blob = np.expand_dims(face, axis=0)
    
    # Verify shape
    expected_shape = (1, 3, CONFIG["recognition"]["face_size"][0], CONFIG["recognition"]["face_size"][1])
    if blob.shape != expected_shape:
        raise ValueError(f"Recognition blob shape {blob.shape} != expected {expected_shape}")
    
    return blob

def generate_embedding(rec_sess, rec_input_name, face_img):
    """Generate face embedding using ArcFace"""
    try:
        blob = preprocess_face(face_img)
        if blob is None:
            raise ValueError("Failed to preprocess face image")
        
        embedding = rec_sess.run(None, {rec_input_name: blob})[0]
        
        # Flatten the embedding
        embedding = embedding.flatten()
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    except Exception as e:
        print(f"ERROR in generate_embedding: {e}")
        print(f"  Face image shape: {face_img.shape}")
        print(f"  Face image dtype: {face_img.dtype}")
        raise

# ============================================================================
# MAIN LOOP
# ============================================================================
def main():
    print("\n" + "=" * 80)
    print(" " * 20 + "RASPBERRY PI FACE DETECTION SENSOR")
    print(" " * 30 + "Phase 2B - With Display")
    print(" " * 25 + "FIXED: Channel Conversion Issue")
    print("=" * 80 + "\n")
    
    # Initialize components
    det_sess, rec_sess, det_input_name, rec_input_name = initialize_models()
    display = initialize_display()
    camera = initialize_camera()
    network_client = initialize_network()
    
    print("\n" + "=" * 60)
    print("SYSTEM READY")
    print("=" * 60)
    print(f"Detection cycle: {CONFIG['camera']['capture_interval']}s")
    print(f"Network: {'✓ Connected' if state.connected else '✗ Disconnected'}")
    print(f"Display: {'✓ Enabled' if display else '✗ Disabled'}")
    print(f"Preview: {'✓ Enabled' if CONFIG['display']['show_preview'] else '✗ Disabled'}")
    print("\nPress Ctrl+C to stop")
    print("=" * 60 + "\n")
    
    last_capture_time = 0
    frame_count = 0
    
    try:
        while True:
            current_time = time.time()
            
            # Capture frame
            try:
                # Try picamera2 first
                if hasattr(camera, 'capture_array'):
                    frame = camera.capture_array()
                    
                    # CRITICAL FIX: Convert BGRA to BGR if needed
                    if len(frame.shape) == 3 and frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                else:
                    # Fallback to OpenCV
                    ret, frame = camera.read()
                    if not ret:
                        print("Failed to capture frame")
                        time.sleep(0.1)
                        continue
            except Exception as e:
                print(f"Camera error: {e}")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            
            # Verify frame is valid
            if frame is None or frame.size == 0:
                print("Invalid frame captured")
                time.sleep(0.1)
                continue
            
            # Check if it's time to detect (4-second cycle)
            if current_time - last_capture_time >= CONFIG["camera"]["capture_interval"]:
                last_capture_time = current_time
                
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Detection #{state.detection_count + 1}")
                print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
                
                # Detect faces
                try:
                    faces = detect_faces(det_sess, det_input_name, frame)
                except Exception as e:
                    print(f"  ✗ Detection failed: {e}")
                    continue
                
                if faces:
                    # Take the first/largest face
                    face = faces[0]
                    bbox = face["bbox"]
                    score = face["score"]
                    
                    print(f"  ✓ Face detected (confidence: {score:.2f})")
                    print(f"    BBox: {bbox}")
                    
                    # Extract face and generate embedding
                    try:
                        face_img = extract_face_aligned(frame, bbox)
                        embedding = generate_embedding(rec_sess, rec_input_name, face_img)
                        
                        print(f"  ✓ Embedding generated (shape: {embedding.shape})")
                        
                        state.detection_count += 1
                        
                        # Send to laptop if connected
                        if state.connected and state.network_client:
                            timestamp = datetime.now().isoformat()
                            
                            if state.network_client.send_embedding(embedding, timestamp):
                                print(f"  ✓ Sent to laptop")
                            else:
                                print(f"  ✗ Failed to send (disconnected?)")
                                state.connected = False
                        else:
                            print(f"  ⚠️ Not connected to laptop")
                        
                        # Draw on preview if enabled
                        if CONFIG["display"]["show_preview"]:
                            x1, y1, x2, y2 = bbox
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Face: {score:.2f}", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"  ✗ Embedding generation failed: {e}")
                        continue
                else:
                    print(f"  ✗ No face detected")
            
            # Show preview window
            if CONFIG["display"]["show_preview"]:
                # Resize for display
                display_frame = cv2.resize(frame, (640, 480))
                
                # Add status overlay
                status_text = f"Status: {'Connected' if state.connected else 'Disconnected'}"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                           (0, 255, 0) if state.connected else (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Detections: {state.detection_count}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.putText(display_frame, f"Matches: {state.match_count}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow(CONFIG["display"]["preview_window"], display_frame)
                
                # Check for ESC key
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    print("\nESC pressed, exiting...")
                    break
            else:
                # Small sleep to prevent CPU spinning
                time.sleep(0.03)
    
    except KeyboardInterrupt:
        print("\n\nCtrl+C detected, shutting down...")
    
    finally:
        # Cleanup
        print("\n[CLEANUP] Stopping camera...")
        if hasattr(camera, 'stop'):
            camera.stop()
        elif hasattr(camera, 'release'):
            camera.release()
        
        print("[CLEANUP] Closing display...")
        if state.display_window:
            state.display_window.stop()
        
        print("[CLEANUP] Disconnecting network...")
        if state.network_client:
            state.network_client.disconnect()
        
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 60)
        print("SHUTDOWN COMPLETE")
        print(f"Total detections: {state.detection_count}")
        print(f"Total matches: {state.match_count}")
        print("=" * 60)

if __name__ == "__main__":
    main()
