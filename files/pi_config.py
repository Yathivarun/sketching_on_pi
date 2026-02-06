"""
Pi Configuration File
Shared settings for all Pi modules
"""

from pathlib import Path

# ============================================================================
# NETWORK CONFIGURATION
# ============================================================================

LAPTOP_IP = "192.168.137.1"
LAPTOP_PORT = 5000
CONNECTION_TIMEOUT = 10  # seconds
RECONNECT_DELAY = 5  # seconds between reconnection attempts

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

CAMERA_ID = 0  # Default Pi camera (try 1, 2 if 0 fails)
CAMERA_WIDTH = 640  # Lower resolution for Pi (not 1280)
CAMERA_HEIGHT = 480  # Lower resolution for Pi (not 720)
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 480

# Camera backend (use different backend if default fails)
# Options: cv2.CAP_V4L2 (Linux), cv2.CAP_DSHOW (Windows), cv2.CAP_ANY (auto)
# For Pi Camera Module: cv2.CAP_V4L2 or cv2.CAP_ANY
CAMERA_BACKEND = None  # None = auto-detect, or specify: cv2.CAP_V4L2

# Camera warmup settings
CAMERA_WARMUP_FRAMES = 30  # Number of frames to discard on startup
CAMERA_WARMUP_DELAY = 0.05  # Delay between warmup frames (seconds)

# ============================================================================
# FACE DETECTION CONFIGURATION
# ============================================================================

# Model paths (InsightFace buffalo_l models)
INSIGHTFACE_MODEL_DIR = Path.home() / ".insightface/models/buffalo_l"
FACE_DETECTOR_MODEL = INSIGHTFACE_MODEL_DIR / "det_500m.onnx"  # SCRFD detector
FACE_RECOGNIZER_MODEL = INSIGHTFACE_MODEL_DIR / "w600k_r50.onnx"  # ArcFace recognizer

# Detection settings
FACE_DET_THRESH = 0.5  # Face detection confidence threshold
FACE_MIN_SIZE = 40  # Minimum face size in pixels

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

DISPLAY_WIDTH = 1920  # Full HD display
DISPLAY_HEIGHT = 1080
DISPLAY_FULLSCREEN = True

# Slideshow settings
SLIDESHOW_INTERVAL = 3.0  # seconds per image
MATCH_DISPLAY_INTERVAL = 5.0  # seconds per matched image
MATCH_DISPLAY_CYCLES = 2  # How many times to cycle through matched images

# Stock images directory for slideshow
STOCK_IMAGES_DIR = Path("stock_images")
STOCK_IMAGES_DIR.mkdir(exist_ok=True)

# Temporary storage for received images
RECEIVED_IMAGES_DIR = Path("received_images")
RECEIVED_IMAGES_DIR.mkdir(exist_ok=True)

# ============================================================================
# PERFORMANCE OPTIMIZATION FOR RASPBERRY PI
# ============================================================================

# ONNX Runtime settings (CPU-optimized for Pi)
ONNX_PROVIDERS = ['CPUExecutionProvider']
ONNX_INTRA_THREADS = 2  # Pi typically has 4 cores, use 2 for inference
ONNX_INTER_THREADS = 1

# Memory management
MAX_CACHED_IMAGES = 50  # Maximum stock images to keep in memory
IMAGE_QUALITY = 85  # JPEG quality for display (balance quality vs memory)

# Threading
USE_THREADING = True  # Enable background processing
