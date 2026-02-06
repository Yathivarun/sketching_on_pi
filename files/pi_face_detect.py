"""
Pi Face Detection & Embedding Module
Handles camera capture, face detection, and embedding generation.
Sends embeddings to laptop server for matching.

OPTIMIZED FOR RASPBERRY PI:
- Efficient model loading (load once, cache)
- Low memory footprint
- CPU-only inference
- Minimal dependencies
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import Optional, Tuple
import sys

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("[ERROR] onnxruntime not installed. Install: pip install onnxruntime")
    ONNX_AVAILABLE = False
    sys.exit(1)

# Import configuration
from pi_config import *
from network_protocol import PiClient

# ============================================================================
# INSIGHTFACE MODEL WRAPPERS (CPU-OPTIMIZED)
# ============================================================================

class SCRFD:
    """
    SCRFD Face Detector (ONNX)
    Optimized for Raspberry Pi CPU
    """
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"[DETECTOR] Loading SCRFD model from {model_path}...")
        
        # CPU-optimized session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = ONNX_INTRA_THREADS
        sess_options.inter_op_num_threads = ONNX_INTER_THREADS
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=ONNX_PROVIDERS
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = tuple(self.input_shape[2:4][::-1])  # (width, height)
        
        print(f"[DETECTOR] ✓ Loaded. Input size: {self.input_size}")
    
    def detect(self, img: np.ndarray, threshold: float = FACE_DET_THRESH):
        """
        Detect faces in image.
        
        Args:
            img: BGR image (OpenCV format)
            threshold: Detection confidence threshold
        
        Returns:
            List of face bboxes and landmarks, or None if no faces
        """
        # Preprocess
        img_resized = cv2.resize(img, self.input_size)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: img_batch})
        
        # Parse outputs (simplified - assumes SCRFD output format)
        # Format: [scores, bboxes, landmarks]
        scores = outputs[0]  # Shape: [N, ]
        bboxes = outputs[1]  # Shape: [N, 4]
        kpss = outputs[2] if len(outputs) > 2 else None  # Shape: [N, 5, 2]
        
        # Filter by threshold
        valid_indices = scores > threshold
        
        if not np.any(valid_indices):
            return None
        
        # Scale bboxes back to original image size
        scale_x = img.shape[1] / self.input_size[0]
        scale_y = img.shape[0] / self.input_size[1]
        
        bboxes_scaled = bboxes[valid_indices].copy()
        bboxes_scaled[:, [0, 2]] *= scale_x
        bboxes_scaled[:, [1, 3]] *= scale_y
        
        if kpss is not None:
            kpss_scaled = kpss[valid_indices].copy()
            kpss_scaled[:, :, 0] *= scale_x
            kpss_scaled[:, :, 1] *= scale_y
        else:
            kpss_scaled = None
        
        return {
            'bboxes': bboxes_scaled,
            'landmarks': kpss_scaled,
            'scores': scores[valid_indices]
        }


class ArcFaceONNX:
    """
    ArcFace Face Recognition Model (ONNX)
    Generates 512D embeddings
    """
    def __init__(self, model_path: str):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"[RECOGNIZER] Loading ArcFace model from {model_path}...")
        
        # CPU-optimized session options
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = ONNX_INTRA_THREADS
        sess_options.inter_op_num_threads = ONNX_INTER_THREADS
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=ONNX_PROVIDERS
        )
        
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_size = (112, 112)  # ArcFace standard input size
        
        print(f"[RECOGNIZER] ✓ Loaded. Embedding size: 512D")
    
    def get_embedding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Generate 512D embedding from aligned face image.
        
        Args:
            face_img: BGR face image (will be resized to 112x112)
        
        Returns:
            512D embedding vector
        """
        # Preprocess (match InsightFace preprocessing)
        face_resized = cv2.resize(face_img, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = (face_rgb.astype(np.float32) - 127.5) / 128.0
        face_transposed = face_normalized.transpose(2, 0, 1)
        face_batch = np.expand_dims(face_transposed, axis=0)
        
        # Run inference
        embedding = self.session.run(None, {self.input_name: face_batch})[0]
        
        # Return as 1D vector
        return embedding.flatten()


# ============================================================================
# FACE ALIGNMENT (Simplified for Pi)
# ============================================================================

def align_face(img: np.ndarray, bbox: np.ndarray, landmarks: np.ndarray = None) -> np.ndarray:
    """
    Extract and align face from image.
    
    Args:
        img: BGR image
        bbox: Face bounding box [x1, y1, x2, y2]
        landmarks: Optional 5-point landmarks
    
    Returns:
        Aligned face image (BGR)
    """
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add padding
    w = x2 - x1
    h = y2 - y1
    pad_w = int(w * 0.2)
    pad_h = int(h * 0.2)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img.shape[1], x2 + pad_w)
    y2 = min(img.shape[0], y2 + pad_h)
    
    # Crop face
    face_crop = img[y1:y2, x1:x2]
    
    # If landmarks provided, use for alignment (simplified)
    if landmarks is not None:
        # For now, just return crop. Full alignment can be added later.
        pass
    
    return face_crop


# ============================================================================
# PI FACE CAPTURE APPLICATION
# ============================================================================

class PiFaceCapture:
    """
    Main application for Pi face detection and embedding generation.
    
    Workflow:
    1. Open camera preview window
    2. User clicks to capture
    3. Detect face in captured image
    4. Generate embedding
    5. Send to laptop
    6. Wait for match result
    """
    
    def __init__(self):
        self.running = False
        self.network_client = None
        
        # Models (loaded once, cached)
        self.detector = None
        self.recognizer = None
        
        # Camera
        self.cap = None
        
        # State
        self.last_capture_time = 0
        self.capture_cooldown = 1.0  # Minimum 1 second between captures
        
        # Match result callback
        self.on_match_result = None
        self.on_images_received = None
    
    def load_models(self):
        """Load face detection and recognition models."""
        print("[INIT] Loading models...")
        
        # Check if models exist
        if not FACE_DETECTOR_MODEL.exists():
            raise FileNotFoundError(
                f"Face detector model not found: {FACE_DETECTOR_MODEL}\n"
                f"Please download InsightFace buffalo_l models to ~/.insightface/models/buffalo_l/"
            )
        
        if not FACE_RECOGNIZER_MODEL.exists():
            raise FileNotFoundError(
                f"Face recognizer model not found: {FACE_RECOGNIZER_MODEL}\n"
                f"Please download InsightFace buffalo_l models to ~/.insightface/models/buffalo_l/"
            )
        
        # Load models
        self.detector = SCRFD(str(FACE_DETECTOR_MODEL))
        self.recognizer = ArcFaceONNX(str(FACE_RECOGNIZER_MODEL))
        
        print("[INIT] ✓ Models loaded successfully")
    
    def connect_to_laptop(self) -> bool:
        """Connect to laptop server."""
        print(f"[NETWORK] Connecting to laptop at {LAPTOP_IP}:{LAPTOP_PORT}...")
        
        self.network_client = PiClient(laptop_ip=LAPTOP_IP, port=LAPTOP_PORT)
        
        # Set callbacks
        self.network_client.on_match_result = self._handle_match_result
        self.network_client.on_images_received = self._handle_images_received
        self.network_client.on_disconnected = self._handle_disconnected
        
        # Attempt connection
        if self.network_client.connect(timeout=CONNECTION_TIMEOUT):
            print("[NETWORK] ✓ Connected to laptop")
            return True
        else:
            print("[NETWORK] ✗ Connection failed")
            return False
    
    def start_camera(self):
        """Initialize camera with robust error handling."""
        print(f"[CAMERA] Opening camera {CAMERA_ID}...")
        
        # Try with specified backend or auto-detect
        if CAMERA_BACKEND is not None:
            self.cap = cv2.VideoCapture(CAMERA_ID, CAMERA_BACKEND)
            print(f"[CAMERA] Using backend: {CAMERA_BACKEND}")
        else:
            self.cap = cv2.VideoCapture(CAMERA_ID)
            print(f"[CAMERA] Using auto-detected backend")
        
        if not self.cap.isOpened():
            print(f"[ERROR] Failed to open camera {CAMERA_ID}")
            print("[CAMERA] Troubleshooting tips:")
            print("  1. Check camera connection: ls /dev/video*")
            print("  2. For Pi Camera Module: sudo raspi-config → Interface → Camera → Enable")
            print("  3. Try different CAMERA_ID (0, 1, 2) in pi_config.py")
            print("  4. Try different CAMERA_BACKEND in pi_config.py")
            raise RuntimeError(f"Failed to open camera {CAMERA_ID}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        # Set buffer size to 1 to get latest frame
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Warm up - read and discard frames to let camera adjust
        print(f"[CAMERA] Warming up ({CAMERA_WARMUP_FRAMES} frames)...")
        successful_reads = 0
        
        for i in range(CAMERA_WARMUP_FRAMES):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                successful_reads += 1
            time.sleep(CAMERA_WARMUP_DELAY)
        
        print(f"[CAMERA] Warmup complete ({successful_reads}/{CAMERA_WARMUP_FRAMES} successful reads)")
        
        if successful_reads == 0:
            self.cap.release()
            raise RuntimeError("Camera warmup failed - no frames could be read")
        
        # Final verification
        ret, test_frame = self.cap.read()
        if not ret or test_frame is None:
            self.cap.release()
            raise RuntimeError("Camera verification failed - cannot read frames")
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[CAMERA] ✓ Camera ready")
        print(f"[CAMERA] Resolution: {actual_width}x{actual_height}")
        
        if actual_width != CAMERA_WIDTH or actual_height != CAMERA_HEIGHT:
            print(f"[CAMERA] ⚠️ Requested {CAMERA_WIDTH}x{CAMERA_HEIGHT} but got {actual_width}x{actual_height}")
            print(f"[CAMERA] This is normal for some cameras")
    
    def run(self):
        """Main application loop."""
        self.running = True
        
        # Create preview window
        window_name = "Pi Face Capture - Press SPACE to capture, Q to quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, PREVIEW_WIDTH, PREVIEW_HEIGHT)
        
        print("\n" + "="*60)
        print("PI FACE CAPTURE READY")
        print("="*60)
        print("Press SPACE to capture image")
        print("Press Q to quit")
        print("="*60 + "\n")
        
        status_text = "Ready - Press SPACE to capture"
        status_color = (0, 255, 0)  # Green
        
        while self.running:
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                print("[ERROR] Failed to read frame")
                break
            
            # Display frame with status overlay
            display_frame = frame.copy()
            
            # Add status text
            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2
            )
            
            # Add connection status
            conn_status = "Connected" if (self.network_client and self.network_client.connected) else "Disconnected"
            conn_color = (0, 255, 0) if (self.network_client and self.network_client.connected) else (0, 0, 255)
            cv2.putText(
                display_frame,
                f"Network: {conn_status}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                conn_color,
                2
            )
            
            cv2.imshow(window_name, display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Q or ESC
                print("[SYSTEM] Quit requested")
                break
            
            elif key == ord(' '):  # SPACE
                # Check cooldown
                current_time = time.time()
                if current_time - self.last_capture_time < self.capture_cooldown:
                    print("[CAPTURE] Please wait before next capture")
                    continue
                
                self.last_capture_time = current_time
                
                # Capture and process
                status_text = "Processing..."
                status_color = (0, 255, 255)  # Yellow
                
                success = self.capture_and_process(frame)
                
                if success:
                    status_text = "Sent to laptop - waiting for result..."
                    status_color = (255, 255, 0)  # Cyan
                else:
                    status_text = "No face detected - try again"
                    status_color = (0, 0, 255)  # Red
        
        # Cleanup
        self.stop()
    
    def capture_and_process(self, frame: np.ndarray) -> bool:
        """
        Capture frame, detect face, generate embedding, send to laptop.
        
        Returns:
            True if successfully sent, False otherwise
        """
        print("\n[CAPTURE] Processing frame...")
        
        # 1. Detect faces
        start_time = time.time()
        detection_result = self.detector.detect(frame)
        
        if detection_result is None or len(detection_result['bboxes']) == 0:
            print("[CAPTURE] ✗ No face detected")
            return False
        
        print(f"[CAPTURE] ✓ Detected {len(detection_result['bboxes'])} face(s) in {time.time()-start_time:.2f}s")
        
        # Use first/best face
        bbox = detection_result['bboxes'][0]
        landmarks = detection_result['landmarks'][0] if detection_result['landmarks'] is not None else None
        score = detection_result['scores'][0]
        
        print(f"[CAPTURE] Face confidence: {score:.3f}")
        
        # 2. Align face
        aligned_face = align_face(frame, bbox, landmarks)
        
        # 3. Generate embedding
        start_time = time.time()
        embedding = self.recognizer.get_embedding(aligned_face)
        print(f"[CAPTURE] ✓ Generated embedding in {time.time()-start_time:.2f}s")
        print(f"[CAPTURE] Embedding shape: {embedding.shape}, sample: {embedding[:3]}")
        
        # 4. Send to laptop
        if self.network_client and self.network_client.connected:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            success = self.network_client.send_embedding(embedding, timestamp)
            
            if success:
                print("[CAPTURE] ✓ Sent to laptop")
                return True
            else:
                print("[CAPTURE] ✗ Failed to send")
                return False
        else:
            print("[CAPTURE] ✗ Not connected to laptop")
            return False
    
    def _handle_match_result(self, msg: dict):
        """Handle match result from laptop."""
        hit = msg.get('hit', False)
        person_id = msg.get('person_id', 'unknown')
        score = msg.get('score', 0.0)
        
        if hit:
            print(f"[MATCH] ✓ HIT: Person #{person_id} (confidence: {score:.3f})")
            print(f"[MATCH] Waiting for images...")
        else:
            print(f"[MATCH] ✗ MISS (best score: {score:.3f})")
        
        # Call external callback if set
        if self.on_match_result:
            self.on_match_result(msg)
    
    def _handle_images_received(self, images: list):
        """Handle images received from laptop."""
        print(f"[IMAGES] ✓ Received {len(images)} images from laptop")
        
        # Call external callback if set
        if self.on_images_received:
            self.on_images_received(images)
    
    def _handle_disconnected(self):
        """Handle disconnection from laptop."""
        print("[NETWORK] ✗ Disconnected from laptop")
        print("[NETWORK] Attempting to reconnect in 5 seconds...")
        
        # Try to reconnect after delay
        time.sleep(RECONNECT_DELAY)
        self.connect_to_laptop()
    
    def stop(self):
        """Cleanup resources."""
        self.running = False
        
        print("[SYSTEM] Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        if self.network_client:
            self.network_client.disconnect()
        
        cv2.destroyAllWindows()
        
        print("[SYSTEM] Shutdown complete")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    try:
        app = PiFaceCapture()
        
        # Load models
        app.load_models()
        
        # Connect to laptop
        app.connect_to_laptop()
        
        # Start camera
        app.start_camera()
        
        # Run main loop
        app.run()
        
    except KeyboardInterrupt:
        print("\n[SYSTEM] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[SYSTEM] Exiting...")


if __name__ == "__main__":
    main()
