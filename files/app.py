"""
Sketch AI - FastAPI Backend with Pi Integration
Connects Camera, Preprocessing, Generation, and Pi Sensor into a web API.
"""

import torch
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
# ----------------------------------------------------
# ----------------------------------------------------

import os
import cv2
import json
import asyncio
import threading
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import Core Modules
import preprocess
import generate
from camera_manager import camera
import base64
import numpy as np
from recognition_manager import recognizer

from collections import deque 

# Configuration
QUEUE_THRESHOLD = 5
SECONDS_PER_IMAGE = 5.0
MIN_DURATION = 10.0 # Minimum time on screen even if only 1 image

# ============================================================================
# CONFIGURATION & STATE
# ============================================================================

INPUTS_DIR = Path("inputs")
BODY_DIR = INPUTS_DIR / "body"
FACES_DIR = INPUTS_DIR / "faces"
PREPROCESSED_DIR = Path("preprocessed_data")
OUTPUTS_DIR = Path("outputs/generated_sketches")

# Ensure directories exist
for d in [BODY_DIR, FACES_DIR, PREPROCESSED_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# GPU Lock to prevent Out-Of-Memory errors
gpu_lock = asyncio.Lock()

# Pydantic Models for API Inputs
class MetadataOverride(BaseModel):
    gender: Optional[str] = None
    has_beard: Optional[bool] = None

class GenerateRequest(BaseModel):
    use_lora: bool = True
    faceid_strength: float = 0.85
    use_lcm: bool = True

class BatchRequest(BaseModel):
    ids: List[str]


# ============================================================================
# PI NETWORK SERVER
# ============================================================================

# Try to import Pi network support
PI_SUPPORT_AVAILABLE = False
try:
    from network_protocol import LaptopServer
    PI_SUPPORT_AVAILABLE = True
    print("[INIT] ✓ Pi network support available")
except ImportError:
    print("[INIT] ⚠️ Pi network support not available (network_protocol.py missing)")
    LaptopServer = None

# Global Pi Server instance
pi_server = None
pi_stats = {
    "connected": False,
    "embeddings_received": 0,
    "matches_sent": 0,
    "last_embedding_time": None,
    "connection_time": None
}

def handle_pi_connected(addr):
    """Called when Pi connects to laptop."""
    global pi_stats
    pi_stats["connected"] = True
    pi_stats["connection_time"] = time.time()
    print(f"[PI] ✓ Pi connected from {addr}")

def handle_pi_disconnected():
    """Called when Pi disconnects."""
    global pi_stats
    pi_stats["connected"] = False
    print("[PI] ✗ Pi disconnected")

def handle_pi_embedding(embedding, timestamp):
    """
    Called when Pi sends an embedding.
    This is the core integration point!
    
    Flow:
    1. Receive embedding from Pi
    2. Match against local DB
    3. If HIT: Trigger display on both laptop + Pi
    4. If MISS: Send "no match" result to Pi
    """
    global pi_stats
    pi_stats["embeddings_received"] += 1
    pi_stats["last_embedding_time"] = time.time()
    
    print(f"[PI] Received embedding at {timestamp}")
    
    # FIX: ensure flat vector
    embedding = np.array(embedding).flatten()
    
    print(f"     Shape: {embedding.shape}, Sample: {embedding[:3]}")
    
    # Match against database
    match_id, score = recognizer.find_match(embedding)
    
    if match_id:
        print(f"[PI] ✓ MATCH: Person #{match_id} (score: {score:.3f})")
        
        # Trigger laptop display (using the same logic as sensor)
        success = activate_poi_logic(match_id)
        
        # Load sketch images to send to Pi
        person_dir = OUTPUTS_DIR / match_id
        scenes = sorted(person_dir.glob("*_scene_*.jpg"))
        
        if scenes and success:
            # Read actual image data to send to Pi
            pi_image_data = []
            for scene_path in scenes:
                with open(scene_path, "rb") as f:
                    pi_image_data.append(f.read())
            
            # Send to Pi
            if pi_server:
                pi_server.send_match_result(hit=True, person_id=match_id, score=score)
                pi_server.send_images(pi_image_data)
                
                pi_stats["matches_sent"] += 1
                print(f"     → Sent {len(pi_image_data)} images to Pi")
        else:
            # No scenes generated yet or display failed
            print(f"     ✗ No scenes found or display failed for #{match_id}")
            if pi_server:
                pi_server.send_match_result(hit=False, score=score)
    else:
        print(f"     ✗ No match (best score: {score:.2f})")
        if pi_server:
            pi_server.send_match_result(hit=False, score=score)


# ============================================================================
# LIFESPAN (STARTUP/SHUTDOWN)
# ============================================================================

import cv2

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pi_server
    
    # --- STARTUP ---
    print("\n" + "="*60)
    print("🚀 SKETCH AI SYSTEM IS ONLINE")
    print("="*60)
    print("📡 DASHBOARD:   http://localhost:8000/admin")
    print("📺 DISPLAY:     http://localhost:8000/display")
    print("👁️ SENSOR:      http://localhost:8000/sensor")
    if PI_SUPPORT_AVAILABLE:
        print("🔌 PI STATUS:   http://localhost:8000/pi_status")
    print("📚 DOCS:        http://localhost:8000/docs")
    print("="*60 + "\n")

    # Start Pi Server if available
    if PI_SUPPORT_AVAILABLE and LaptopServer:
        print("[PI] Starting Pi network server...")
        pi_server = LaptopServer()
        pi_server.on_client_connected = handle_pi_connected
        pi_server.on_client_disconnected = handle_pi_disconnected
        pi_server.on_embedding_received = handle_pi_embedding
        
        if pi_server.start():
            print("[PI] ✓ Server started on port 5000")
            print("[PI] Waiting for Pi to connect...")
        else:
            print("[PI] ✗ Failed to start Pi server")
    else:
        print("[PI] Pi network support disabled (network_protocol.py not found)")

    # Warmup
    async with gpu_lock:
        pass 
            
    yield  # <--- Application runs here
    
    # --- SHUTDOWN ---
    print("\n[SYSTEM] Shutdown initiated...")

    # 1. Stop Pi Server
    if pi_server:
        print("[SYSTEM] Stopping Pi server...")
        pi_server.stop()

    # 2. Stop Camera
    if camera.is_running:
        print("[SYSTEM] Stopping camera...")
        camera.stop()

    # 3. Force Close OpenCV Windows
    import cv2
    print("[SYSTEM] Destroying GUI windows...")
    cv2.destroyAllWindows()
    
    for i in range(5):
        cv2.waitKey(1)
        
    print("[SYSTEM] Shutdown complete.")

app = FastAPI(title="Sketch AI API", lifespan=lifespan)

# In Configuration Section
app.state.poi_detection_enabled = True # Master switch

# ============================================================================
# STATIC MOUNTS
# ============================================================================

# Serve Generated Results
app.mount("/results", StaticFiles(directory=OUTPUTS_DIR), name="results")

# Serve Input Images (for Admin Grid)
app.mount("/static/inputs", StaticFiles(directory=INPUTS_DIR), name="inputs")

# Serve UI Assets (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_identity_status(person_id: str):
    """Checks file existence to determine status."""
    return {
        "id": person_id,
        "has_body": (BODY_DIR / f"{person_id}.jpg").exists(),
        "has_faces": len(list(FACES_DIR.glob(f"{person_id}-*.jpg"))) > 0,
        "preprocessed": (PREPROCESSED_DIR / person_id / "metadata.json").exists(),
        "generated": (OUTPUTS_DIR / person_id / f"{person_id}-sketch.jpg").exists(),
    }

def activate_poi_logic(person_id: str):
    """
    Internal logic to switch the display to a specific person immediately.
    STRICT MODE: Only shows Scene Compositions. Ignores raw sketches.
    """
    # 1. Find Images
    person_dir = OUTPUTS_DIR / person_id
    images = []
    
    if person_dir.exists():
        scene_files = list(person_dir.glob("*_scene_*.jpg"))
        
        # Sort to ensure consistent playback order (scene1, scene2, ...)
        scene_files.sort()
        
        # Convert to Web URLs
        images = [f"/results/{person_id}/{f.name}" for f in scene_files]
        
    # 2. Validation
    if not images:
        # If no scenes are found (even if a sketch exists), we return False.
        # This prevents the "Sketch First" flash.
        # The sensor will just try again in 1 second.
        return False

    # 3. Calculate Dynamic Duration (5s per image, min 10s)
    calc_duration = len(images) * SECONDS_PER_IMAGE
    final_duration = max(MIN_DURATION, calc_duration)

    # 4. Update State
    display_state.current_mode = "poi"
    display_state.current_poi_id = person_id
    display_state.current_poi_images = images
    display_state.poi_start_time = time.time()
    display_state.poi_duration = final_duration

    print(f"[DISPLAY] Showing #{person_id} for {final_duration}s ({len(images)} scenes)")
    return True

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {"message": "Sketch AI API is running. Go to /admin for dashboard."}

@app.get("/admin", response_class=HTMLResponse)
async def admin_panel(request: Request):
    """Serves the Admin Dashboard."""
    return templates.TemplateResponse("admin.html", {"request": request})

@app.get("/docs/professional", response_class=HTMLResponse)
async def professional_docs(request: Request):
    """Serves the Professional Documentation."""
    return templates.TemplateResponse("docs.html", {"request": request})

# --- DATA MANAGEMENT ---

@app.get("/api/identities")
async def list_identities():
    """Scans folders to list all registered people."""
    identities = []
    # Scan body images as the source of truth
    files = sorted(list(BODY_DIR.glob("*.jpg")))
    
    for f in files:
        person_id = f.stem
        status = get_identity_status(person_id)
        
        # Load metadata if available (to show current gender overrides)
        meta_path = PREPROCESSED_DIR / person_id / "metadata.json"
        metadata = {}
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as mf:
                    metadata = json.load(mf)
            except:
                pass
        
        status["metadata"] = metadata
        identities.append(status)
        
    return identities

@app.post("/api/session/clear")
async def clear_session():
    """Moves current inputs to an 'archive' folder to reset the demo view."""
    # (Implementation stub: For now, we can just return success or actually move files)
    # A true implementation would move files to 'inputs/archive/{timestamp}'
    print("[ADMIN] Session Clear Requested")
    return {"status": "success", "message": "Session cleared (Simulated)"}

# --- CAMERA & REGISTRATION ---

@app.post("/api/camera/toggle")
async def toggle_camera(state: bool):
    """Manually turns the camera ON or OFF for the preview feed."""
    if state:
        if not camera.is_running:
            camera.start()
        return {"status": "on"}
    else:
        if camera.is_running:
            camera.stop()
        return {"status": "off"}

def generate_video_stream():
    """Generator function for MJPEG streaming."""
    # Only stream if camera is actually running
    while True:
        if not camera.is_running:
             # Return a black frame or simple wait if camera is off
            time.sleep(0.5)
            continue
            
        frame_bytes = camera.get_frame()
        if frame_bytes:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.1)

@app.get("/video_feed")
async def video_feed():
    """Live video stream for the Admin Panel."""
    return StreamingResponse(generate_video_stream(), 
                            media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/api/register/live")
async def register_live_identity():
    """
    MANUAL REGISTRATION MODE:
    1. Pauses/Ensures camera is available for exclusive access.
    2. Opens a server-side popup window.
    3. Admin presses SPACE to capture Body, then Front, Left, Right faces.
    4. SHUTS DOWN camera after capture (as requested).
    """
    
    # 1. Stop the background stream manager to take full control
    if camera.is_running:
        print("[REGISTER] Stopping background stream for exclusive access...")
        camera.stop()
        time.sleep(1) 

    # 2. Get new ID
    existing_ids = [int(f.stem) for f in BODY_DIR.glob("*.jpg") if f.stem.isdigit()]
    new_id = str(max(existing_ids) + 1) if existing_ids else "1001"

    print(f"[REGISTER] Starting live capture for ID #{new_id}")

    # 3. Open live capture window (BLOCKING function call)
    try:
        success = camera_capture_sequence(new_id)
        
        if success:
            return {
                "status": "success",
                "id": new_id,
                "message": f"Identity {new_id} registered!"
            }
        else:
            raise HTTPException(500, "User canceled or capture failed")
    except Exception as e:
        print(f"[REGISTER] Error: {e}")
        raise HTTPException(500, str(e))

def camera_capture_sequence(person_id: str) -> bool:
    """
    Server-side camera capture sequence.
    Creates a temporary capture window and waits for user input.
    Returns True if successful, False if cancelled.
    """
    print(f"[CAPTURE] Opening camera for ID #{person_id}...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return False
    
    # Wait for camera warmup
    for _ in range(10):
        cap.read()
    
    # State tracking
    stage = 0  # 0=body, 1=front, 2=left, 3=right
    stage_names = ["BODY (Full View)", "FACE (Front)", "FACE (Left 45°)", "FACE (Right 45°)"]
    captured_frames = {}
    
    window_name = f"Registration #{person_id} - Press SPACE to capture, ESC to cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display overlay
        display = frame.copy()
        
        if stage < len(stage_names):
            instruction = stage_names[stage]
            cv2.putText(display, instruction, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(display, f"Step {stage+1}/4 - Press SPACE", (20, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(display, "All Captured! Saving...", (20, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:  # ESC
            print("[CAPTURE] Cancelled by user")
            cap.release()
            cv2.destroyWindow(window_name)
            return False
        
        elif key == ord(' '):  # SPACE
            if stage < len(stage_names):
                captured_frames[stage] = frame.copy()
                print(f"[CAPTURE] ✓ Captured: {stage_names[stage]}")
                stage += 1
                
                if stage == len(stage_names):
                    # All captured, save and exit
                    break
    
    cap.release()
    cv2.destroyWindow(window_name)
    
    # Save captured images
    try:
        # Body
        body_path = BODY_DIR / f"{person_id}.jpg"
        cv2.imwrite(str(body_path), captured_frames[0])
        print(f"[SAVE] Body: {body_path}")
        
        # Faces
        face_labels = ["front", "left", "right"]
        for i, label in enumerate(face_labels):
            face_path = FACES_DIR / f"{person_id}-{label}.jpg"
            cv2.imwrite(str(face_path), captured_frames[i+1])
            print(f"[SAVE] Face ({label}): {face_path}")
        
        print(f"[CAPTURE] ✓ All images saved for #{person_id}")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to save images: {e}")
        return False


# --- PREPROCESSING ---

@app.post("/api/preprocess/{person_id}")
async def preprocess_person(person_id: str, override: MetadataOverride = None):
    """Processes a person's photos to prepare for sketch generation."""
    body_path = BODY_DIR / f"{person_id}.jpg"
    
    if not body_path.exists():
        raise HTTPException(404, f"Body image not found for {person_id}")
    
    try:
        # Convert Pydantic model to dict (if provided)
        override_dict = None
        if override and (override.gender or override.has_beard is not None):
            override_dict = {}
            if override.gender:
                override_dict['gender'] = override.gender
            if override.has_beard is not None:
                override_dict['has_beard'] = override.has_beard
        
        # Run preprocessing
        output_dir = preprocess.preprocess_image(
            str(body_path),
            output_id=person_id,
            metadata_override=override_dict
        )
        
        # Reload recognition DB to include new embedding
        recognizer.reload_db()
        
        return {
            "status": "success",
            "id": person_id,
            "output_dir": output_dir
        }
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        raise HTTPException(500, str(e))


# --- GENERATION ---

@app.post("/api/generate/{person_id}")
async def generate_sketch(person_id: str, req: GenerateRequest = None):
    """
    Generates a portrait sketch for a preprocessed person.
    Supports optional parameter override.
    """
    # Check if preprocessed
    meta_path = PREPROCESSED_DIR / person_id / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(404, f"Person {person_id} not preprocessed yet")
    
    # Load metadata
    with open(meta_path, 'r') as f:
        metadata = json.load(f)
    
    # Override defaults with request if provided
    use_lora = req.use_lora if req else True
    faceid_strength = req.faceid_strength if req else 0.85
    use_lcm = req.use_lcm if req else True
    
    try:
        async with gpu_lock:
            result = generate.generate_sketch(
                person_id=person_id,
                use_lora=use_lora,
                faceid_strength=faceid_strength,
                use_lcm=use_lcm,
                metadata=metadata
            )
        
        return {
            "status": "success",
            "id": person_id,
            "sketch_path": result["sketch_path"],
            "scenes": result.get("scenes", [])
        }
    except Exception as e:
        print(f"[ERROR] Generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.post("/api/batch/preprocess")
async def batch_preprocess(request: BatchRequest):
    """Batch preprocess multiple people."""
    results = []
    
    for person_id in request.ids:
        try:
            body_path = BODY_DIR / f"{person_id}.jpg"
            if not body_path.exists():
                results.append({"id": person_id, "status": "skipped", "reason": "No body image"})
                continue
            
            output_dir = preprocess.preprocess_image(str(body_path), output_id=person_id)
            results.append({"id": person_id, "status": "success", "output_dir": output_dir})
        except Exception as e:
            results.append({"id": person_id, "status": "error", "error": str(e)})
    
    # Reload DB once after all preprocessing
    recognizer.reload_db()
    
    return {"results": results}


@app.post("/api/batch/generate")
async def batch_generate(request: BatchRequest, req: GenerateRequest = None):
    """Batch generate sketches for multiple people."""
    results = []
    
    use_lora = req.use_lora if req else True
    faceid_strength = req.faceid_strength if req else 0.85
    use_lcm = req.use_lcm if req else True
    
    for person_id in request.ids:
        try:
            meta_path = PREPROCESSED_DIR / person_id / "metadata.json"
            if not meta_path.exists():
                results.append({"id": person_id, "status": "skipped", "reason": "Not preprocessed"})
                continue
            
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            
            async with gpu_lock:
                result = generate.generate_sketch(
                    person_id=person_id,
                    use_lora=use_lora,
                    faceid_strength=faceid_strength,
                    use_lcm=use_lcm,
                    metadata=metadata
                )
            
            results.append({
                "id": person_id,
                "status": "success",
                "sketch_path": result["sketch_path"],
                "scenes": result.get("scenes", [])
            })
        except Exception as e:
            results.append({"id": person_id, "status": "error", "error": str(e)})
    
    return {"results": results}


# ============================================================================
# DISPLAY SYSTEM (Queue + POI Logic)
# ============================================================================

class PersonQueue:
    def __init__(self, max_size=QUEUE_THRESHOLD):
        self.queue = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, person_id: str):
        """Add person to queue if not already present."""
        if person_id not in self.queue:
            self.queue.append(person_id)
            return True
        return False

    def pop(self):
        """Remove and return the next person, or None if empty."""
        return self.queue.popleft() if self.queue else None

    def contains(self, person_id: str):
        """Check if person is already in queue."""
        return person_id in self.queue

    def is_full(self):
        """Check if queue is at capacity."""
        return len(self.queue) >= self.max_size

    def __len__(self):
        return len(self.queue)

class DisplayState:
    def __init__(self):
        self.current_mode = "slideshow"  # "slideshow" or "poi"
        self.current_poi_id = None
        self.current_poi_images = []
        self.poi_start_time = 0
        self.poi_duration = 0
        self.queue = PersonQueue()

display_state = DisplayState()

@app.post("/api/display/trigger/{person_id}")
async def trigger_poi(person_id: str):
    """
    Admin Manual Trigger.
    Forces the display to show a specific person immediately, BYPASSING the queue.
    """
    # Use the shared helper function logic
    success = activate_poi_logic(person_id)
    
    if success:
        # If manual trigger works, we should clear the queue to prevent confusion
        # (Optional: depends if you want to wipe the playlist or keep it)
        # display_state.queue.queue.clear() 
        
        return {
            "status": "success", 
            "mode": "poi", 
            "message": f"Forced display of #{person_id}"
        }
    else:
        raise HTTPException(404, "No generated images found for this ID")

@app.get("/api/display/status")
async def get_display_status():
    """
    Called by the Display Client every ~1-2 seconds.
    Handles auto-transition logic (Handover).
    """
    # Check if we are in POI mode and if time has expired
    if display_state.current_mode == "poi":
        elapsed = time.time() - display_state.poi_start_time
        
        if elapsed > display_state.poi_duration:
            # --- TIME IS UP! CHECK QUEUE ---
            next_person = display_state.queue.pop()
            
            if next_person:
                # HANDOVER: Queue has someone, show them next
                success = activate_poi_logic(next_person)
                if not success:
                    # If activation failed (no images), try next or revert
                    display_state.current_mode = "slideshow"
                    display_state.current_poi_id = None
            else:
                # QUEUE EMPTY: Go back to sleep (slideshow)
                display_state.current_mode = "slideshow"
                display_state.current_poi_id = None
                display_state.current_poi_images = []
                print("[DISPLAY] Queue empty. Reverting to Slideshow.")

    return {
        "mode": display_state.current_mode,
        "images": display_state.current_poi_images,
        "poi_id": display_state.current_poi_id,
        "queue_len": len(display_state.queue) # Useful for debug
    }

@app.get("/api/display/slideshow-image")
async def get_slideshow_image():
    """Returns a random SCENE from previous generations."""
    import random
    
    # Recursively find all scene images in the outputs folder
    all_scenes = list(OUTPUTS_DIR.glob("**/*_scene_*.jpg"))
    
    if not all_scenes:
        # Fallback to sketches if no scenes exist yet
        all_scenes = list(OUTPUTS_DIR.glob("**/*-sketch.jpg"))

    if not all_scenes:
        return {"url": None} 
        
    choice = random.choice(all_scenes)
    person_id = choice.parent.name
    filename = choice.name
    return {"url": f"/results/{person_id}/{filename}"}

@app.get("/display", response_class=HTMLResponse)
async def display_page(request: Request):
    return templates.TemplateResponse("display.html", {"request": request})

@app.get("/sensor", response_class=HTMLResponse)
async def sensor_page(request: Request):
    """Serves the Sensor Node interface."""
    return templates.TemplateResponse("sensor.html", {"request": request})

class RecognitionRequest(BaseModel):
    image: str  # Base64 encoded image

@app.post("/api/admin/toggle-poi")
async def toggle_poi_detection(enabled: bool):
    """Admin switch to turn on/off the sensor logic."""
    app.state.poi_detection_enabled = enabled
    print(f"[ADMIN] POI Detection set to: {enabled}")
    return {"status": "success", "enabled": enabled}

@app.post("/api/recognize")
async def recognize_face(payload: RecognitionRequest):
    if not app.state.poi_detection_enabled:
        return {"status": "disabled", "message": "Sensor Disabled by Admin"}

    try:
        # 1. Decode Image
        img_str = payload.image
        if "," in img_str: img_str = img_str.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 2. Get Face Details (Embedding + BBox)
        face_data = preprocess.get_live_face_details(frame)

        if not face_data:
            return {"status": "no_face", "message": "Searching..."}

        # 3. Match Logic
        match_id, score = recognizer.find_match(face_data["embedding"])
        bbox = face_data["bbox"]

        # 4. Decision Tree
        if match_id:
            # CHECK A: Is this person ALREADY on screen?
            if display_state.current_poi_id == match_id:
                return {
                    "status": "ignored_active",
                    "id": match_id,
                    "score": score,
                    "bbox": bbox,
                    "message": "Active on Screen"
                }

            # CHECK B: Is this person ALREADY waiting in the queue?
            if display_state.queue.contains(match_id):
                 return {
                    "status": "ignored_queued",
                    "id": match_id,
                    "score": score,
                    "bbox": bbox,
                    "message": "Already in Queue"
                }

            # CHECK C: Is the Screen Idle (Slideshow Mode)?
            # If Idle -> Trigger Immediately (Skip Queue)
            if display_state.current_mode == "slideshow":
                success = activate_poi_logic(match_id)
                if success:
                    return {
                        "status": "triggered",
                        "id": match_id,
                        "score": score,
                        "bbox": bbox,
                        "message": "🚀 Triggered!"
                    }
            
            # CHECK D: Screen is Busy -> Try to Add to Queue
            if display_state.queue.is_full():
                return {
                    "status": "rejected_full",
                    "id": match_id,
                    "score": score,
                    "bbox": bbox,
                    "message": f"Queue Full ({len(display_state.queue)}/{QUEUE_THRESHOLD})"
                }
            else:
                display_state.queue.add(match_id)
                pos = len(display_state.queue)
                return {
                    "status": "queued",
                    "id": match_id,
                    "score": score,
                    "bbox": bbox,
                    "message": f"Added to Queue (#{pos})"
                }
            
        else:
            # SCENARIO: UNKNOWN PERSON
            return {
                "status": "unknown",
                "score": score, 
                "bbox": bbox,
                "message": "Unknown Face"
            }

    except Exception as e:
        print(f"[ERROR] Recog: {e}")
        return {"status": "error", "message": str(e)}


# ============================================================================
# PI NETWORK ENDPOINTS
# ============================================================================

@app.get("/pi_status", response_class=HTMLResponse)
async def pi_status_page(request: Request):
    """Serves the Pi monitoring page."""
    if not PI_SUPPORT_AVAILABLE:
        return HTMLResponse(content="<h1>Pi Support Not Available</h1><p>network_protocol.py not found</p>", status_code=503)
    return templates.TemplateResponse("pi_sensor.html", {"request": request})

@app.get("/api/pi/status")
async def get_pi_status():
    """Get current Pi connection status and statistics."""
    if not PI_SUPPORT_AVAILABLE or not pi_server:
        return {
            "connected": False,
            "embeddings_received": 0,
            "matches_sent": 0,
            "last_embedding_time": None,
            "uptime": 0
        }
    
    uptime = 0
    if pi_stats["connected"] and pi_stats["connection_time"]:
        uptime = time.time() - pi_stats["connection_time"]
    
    return {
        "connected": pi_stats["connected"],
        "embeddings_received": pi_stats["embeddings_received"],
        "matches_sent": pi_stats["matches_sent"],
        "last_embedding_time": pi_stats["last_embedding_time"],
        "uptime": uptime
    }

# ------------------------------------------------------

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    # Run with reload for development
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
