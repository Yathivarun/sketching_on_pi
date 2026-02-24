import os
os.environ["ORT_DISABLE_ALL_LOGGING"] = "1"

import sys
sys.stderr = open(os.devnull, "w")

import cv2
import numpy as np
import onnxruntime as ort
import time
import subprocess
import threading
import queue
import requests
from pathlib import Path
from scrfd import SCRFD

# ---------------- CONFIG ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

STOCK_DIR = Path("stock_images")

RECOGNITION_SERVER = "http://192.168.1.100:8000"   # CHANGE THIS
MINIO_SERVER       = "http://192.168.1.101:9000"   # CHANGE THIS

MIN_FACE_SIZE      = 80
PROCESS_INTERVAL   = 1.5
BURST_FRAMES       = 5
PRECHECK_DOWNSCALE = 0.5
SLIDE_DURATION     = 3
FADE_DURATION      = 0.4
WINDOW_NAME        = "Display"
COOLDOWN_TIME      = 20.0
REQUEST_TIMEOUT    = 5        # seconds for any HTTP call

# ---------------- SCREEN RESOLUTION ----------------
def get_screen_resolution():
    try:
        output = subprocess.check_output(
            "xrandr | grep '*' | head -n 1", shell=True
        ).decode()
        res = output.split()[0]
        return tuple(map(int, res.split('x')))
    except:
        return (1920, 1080)

SCREEN_W, SCREEN_H = get_screen_resolution()

# ---------------- LOAD MODELS ----------------
detector = SCRFD(SCRFD_MODEL)
detector.prepare(-1)

opts = ort.SessionOptions()
rec_sess = ort.InferenceSession(
    ARCFACE_MODEL, opts,
    providers=["CPUExecutionProvider"]
)
rec_input = rec_sess.get_inputs()[0].name

# ---------------- CAMERA ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# ---------------- DISPLAY ----------------
time.sleep(2)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, SCREEN_W, SCREEN_H)
cv2.moveWindow(WINDOW_NAME, 0, 0)

fullscreen = True
cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

# ---------------- THREAD-SAFE QUEUE ----------------
# maxsize=1 — only the latest identity result matters,
# older pending results are discarded automatically.
image_queue = queue.Queue(maxsize=1)

# ---------------- STATE ----------------
current_images   = []
current_index    = 0
last_slide_time  = 0
display_identity = "STOCK"
last_seen_times  = {}
bg_pending       = False      # True while background thread is working

# ---------------- EMBEDDING FUNCTIONS (UNCHANGED) ----------------
def preprocess_rec(face_bgr):
    img = cv2.resize(face_bgr, (112, 112))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = (img.astype(np.float32) - 127.5) * 0.0078125
    img = img.transpose(2, 0, 1)
    return img[None, ...]

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def get_embedding(face_bgr):
    emb = rec_sess.run(None, {rec_input: preprocess_rec(face_bgr)})[0][0]
    return l2_normalize(emb.astype(np.float32))

# ---------------- FACE UTILS (UNCHANGED) ----------------
def valid_face(bbox, shape):
    x1, y1, x2, y2 = bbox[:4].astype(int)
    h, w = shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
        return None
    return (x1, y1, x2, y2)

# ---------------- DISPLAY UTILS (UNCHANGED) ----------------
def resize_fit(img, w=SCREEN_W, h=SCREEN_H):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    resized = cv2.resize(img, (int(iw * scale), int(ih * scale)))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = (h - resized.shape[0]) // 2
    x = (w - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas

def overlay_id(img, text):
    overlay = img.copy()
    cv2.rectangle(overlay,
                  (int(SCREEN_W * 0.73), int(SCREEN_H * 0.93)),
                  (int(SCREEN_W * 0.99), SCREEN_H),
                  (0, 0, 0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.putText(img, text,
                (int(SCREEN_W * 0.74), int(SCREEN_H * 0.97)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 0, 0), 2, cv2.LINE_AA)
    return img

def fade(img1, img2):
    steps = int(FADE_DURATION * 60)
    for i in range(steps + 1):
        alpha = i / steps
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        cv2.imshow(WINDOW_NAME, blended)
        if cv2.waitKey(1) == 27:
            break

# ---------------- LOCAL STOCK LOADER (UNCHANGED LOGIC) ----------------
def load_stock_images():
    images = []
    for img_path in sorted(STOCK_DIR.glob("*")):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)
    return images if images else [np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)]

# ---------------- SERVER CALLS ----------------
def identify_on_server(embedding):
    """Send embedding to recognition server. Returns identity string or 'UNKNOWN'."""
    try:
        r = requests.post(
            f"{RECOGNITION_SERVER}/identify_by_face",
            json={"embedding": embedding.tolist()},
            timeout=REQUEST_TIMEOUT
        )
        data = r.json()
        if data.get("visitor"):
            return str(data["visitor"]["id"])
        return "UNKNOWN"
    except Exception as e:
        print("Recognition server error:", e)
        return "UNKNOWN"

def fetch_minio_urls(visit_uuid):
    """
    Fetch signed image URLs for a visit from the WPU API.
    Endpoint : GET /api/v1/wpu/images?visit_uuid=<uuid>
    Response : {"urls": ["http://...", ...]}
    Returns list of URL strings, empty list on failure.
    """
    try:
        r = requests.get(
            f"{MINIO_SERVER}/api/v1/wpu/images",
            params={"visit_uuid": visit_uuid},
            timeout=REQUEST_TIMEOUT
        )
        data = r.json()
        return data.get("urls", [])
    except Exception as e:
        print("WPU image fetch error:", e)
        return []

def download_image_to_memory(url):
    """Download a single image URL into a numpy array (no disk write)."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        img_array = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img  # None if decode fails
    except Exception as e:
        print(f"Image download error ({url}):", e)
        return None

# ---------------- BACKGROUND WORKER ----------------
def background_identify_and_fetch(embedding):
    """
    Runs in a background thread.
    1. Sends embedding to recognition server
    2. Fetches image URLs from MinIO
    3. Downloads all images into memory
    4. Pushes result onto image_queue for main loop to consume
    Falls back to stock images at any failure point.
    """
    global bg_pending

    identity = identify_on_server(embedding)

    if identity == "UNKNOWN":
        bg_pending = False
        return  # Do not push anything — main loop keeps showing current content

    urls = fetch_minio_urls(identity)

    if urls:
        images = [img for img in (download_image_to_memory(u) for u in urls) if img is not None]
    else:
        images = []

    if not images:
        print(f"No images for {identity}, falling back to stock")
        images = load_stock_images()
        identity = "STOCK"

    # Discard stale result if queue already has one pending
    try:
        image_queue.put_nowait({"identity": identity, "images": images})
    except queue.Full:
        # A newer result is already waiting — discard this one
        pass

    bg_pending = False

# ---------------- INIT ----------------
last_process_time = 0
current_images    = load_stock_images()

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # -------- Consume queue from background thread (thread-safe) --------
    try:
        result = image_queue.get_nowait()
        new_identity = result["identity"]
        new_images   = result["images"]

        time_since_last = now - last_seen_times.get(new_identity, 0)

        # Respect cooldown even for background results
        if new_identity != display_identity and time_since_last > COOLDOWN_TIME:
            prev_img       = resize_fit(current_images[current_index])
            display_identity = new_identity
            current_images   = new_images
            current_index    = 0
            last_slide_time  = now
            next_img = resize_fit(current_images[0])
            fade(prev_img, next_img)

    except queue.Empty:
        pass

    # -------- Face Detection + Embedding (main loop, fast precheck only) --------
    if now - last_process_time > PROCESS_INTERVAL and not bg_pending:
        last_process_time = now

        small = cv2.resize(frame, (0, 0),
                           fx=PRECHECK_DOWNSCALE,
                           fy=PRECHECK_DOWNSCALE)

        bboxes, _ = detector.detect(small, thresh=0.5, input_size=(320, 320))

        if bboxes is not None:
            faces = []

            for _ in range(BURST_FRAMES):
                ret, burst = cap.read()
                if not ret:
                    continue

                bboxes, _ = detector.detect(burst, thresh=0.3, input_size=(640, 640))
                if bboxes is None:
                    continue

                for bbox in bboxes:
                    coords = valid_face(bbox, burst.shape)
                    if coords is None:
                        continue

                    x1, y1, x2, y2 = coords
                    score = float(bbox[4])
                    area  = (x2 - x1) * (y2 - y1)
                    faces.append((score * area, burst[y1:y2, x1:x2]))

            if faces:
                _, best_face = max(faces, key=lambda x: x[0])
                emb = get_embedding(best_face)

                # Fire background thread — never blocks main loop
                bg_pending = True
                t = threading.Thread(
                    target=background_identify_and_fetch,
                    args=(emb,),
                    daemon=True
                )
                t.start()

    # -------- Slideshow (UNCHANGED logic) --------
    if now - last_slide_time > SLIDE_DURATION:
        last_slide_time = now
        prev = resize_fit(current_images[current_index])

        current_index += 1

        if display_identity != "STOCK" and current_index >= len(current_images):
            display_identity = "STOCK"
            current_images   = load_stock_images()
            current_index    = 0
        else:
            current_index = current_index % len(current_images)

        next_img = resize_fit(current_images[current_index])
        fade(prev, next_img)

    display_img = resize_fit(current_images[current_index])
    label       = f"ID: {display_identity}"
    display_img = overlay_id(display_img, label)

    cv2.imshow(WINDOW_NAME, display_img)

    key = cv2.waitKey(1) & 0xFF

    if key in (27, ord('q')):
        break

    if key == ord('f'):
        fullscreen = not fullscreen
        cv2.setWindowProperty(
            WINDOW_NAME,
            cv2.WND_PROP_FULLSCREEN,
            cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        )

cap.release()
cv2.destroyAllWindows()
