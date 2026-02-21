import os
os.environ["ORT_DISABLE_ALL_LOGGING"] = "1"

import sys
sys.stderr = open(os.devnull, "w")

import cv2
import numpy as np
import onnxruntime as ort
import time
from pathlib import Path
from scrfd import SCRFD

# ---------------- CONFIG ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

DB_DIR = "DB/embeddings"
STOCK_DIR = Path("stock_images")
SKETCH_DIR = Path("sketches")

MATCH_THRESHOLD = 1.2
MIN_FACE_SIZE = 80
PROCESS_INTERVAL = 1.5
BURST_FRAMES = 5
PRECHECK_DOWNSCALE = 0.5
SLIDE_DURATION = 3
FADE_DURATION = 0.4
WINDOW_NAME = "Display"
COOLDOWN_TIME = 20.0  # Seconds to wait before re-triggering the same person's slideshow

# ---------------- LOAD MODELS ----------------
detector = SCRFD(SCRFD_MODEL)
detector.prepare(-1)

opts = ort.SessionOptions()
rec_sess = ort.InferenceSession(ARCFACE_MODEL, opts, providers=["CPUExecutionProvider"])
rec_input = rec_sess.get_inputs()[0].name

# ---------------- LOAD DB ----------------
db_embeddings = []
db_ids = []

for file in os.listdir(DB_DIR):
    if file.endswith(".npy"):
        db_embeddings.append(np.load(os.path.join(DB_DIR, file)))
        db_ids.append(os.path.splitext(file)[0])

db_embeddings = np.array(db_embeddings)

# ---------------- CAMERA ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# ---------------- DISPLAY ----------------
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
fullscreen = True
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

current_images = []
current_index = 0
last_slide_time = 0

display_identity = "STOCK"
last_seen_times = {} 

# ---------------- FUNCTIONS ----------------
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

def match_embedding(embedding):
    distances = np.linalg.norm(db_embeddings - embedding, axis=1)
    idx = np.argmin(distances)
    if distances[idx] < MATCH_THRESHOLD:
        return db_ids[idx]
    return "UNKNOWN"

def load_images_for(identity):
    if identity == "UNKNOWN" or identity == "STOCK":
        folder = STOCK_DIR
    else:
        folder = SKETCH_DIR / identity
        if not folder.exists():
            folder = STOCK_DIR

    images = []
    for img_path in sorted(folder.glob("*")):
        img = cv2.imread(str(img_path))
        if img is not None:
            images.append(img)

    return images if images else [np.zeros((1080, 1920, 3), dtype=np.uint8)]

def resize_fit(img, w=1920, h=1080):
    ih, iw = img.shape[:2]
    scale = min(w/iw, h/ih)
    resized = cv2.resize(img, (int(iw*scale), int(ih*scale)))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y = (h - resized.shape[0]) // 2
    x = (w - resized.shape[1]) // 2
    canvas[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return canvas

def overlay_id(img, text):
    overlay = img.copy()
    cv2.rectangle(overlay, (1400, 1000), (1900, 1070), (0,0,0), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.putText(img, text, (1420, 1050),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0,0,0), 2, cv2.LINE_AA)
    return img

def fade(img1, img2):
    steps = int(FADE_DURATION * 60)
    for i in range(steps+1):
        alpha = i / steps
        blended = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
        cv2.imshow(WINDOW_NAME, blended)
        if cv2.waitKey(1) == 27:
            break

# ---------------- LOOP ----------------
last_process_time = 0

current_images = load_images_for("STOCK")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    # -------- Recognition --------
    if now - last_process_time > PROCESS_INTERVAL:
        last_process_time = now

        small = cv2.resize(frame, (0,0),
                           fx=PRECHECK_DOWNSCALE,
                           fy=PRECHECK_DOWNSCALE)

        bboxes, _ = detector.detect(small, thresh=0.5, input_size=(320,320))

        if bboxes is not None:
            faces = []

            for _ in range(BURST_FRAMES):
                ret, burst = cap.read()
                if not ret:
                    continue

                bboxes, _ = detector.detect(burst, thresh=0.3, input_size=(640,640))
                if bboxes is None:
                    continue

                for bbox in bboxes:
                    coords = valid_face(bbox, burst.shape)
                    if coords is None:
                        continue

                    x1,y1,x2,y2 = coords
                    score = float(bbox[4])
                    area = (x2-x1)*(y2-y1)
                    faces.append((score*area, burst[y1:y2, x1:x2]))

            if faces:
                _, best_face = max(faces, key=lambda x: x[0])
                emb = get_embedding(best_face)
                identity = match_embedding(emb)
            else:
                identity = "UNKNOWN"
        else:
            identity = "UNKNOWN"

        if identity != "UNKNOWN":
            # Check how long it has been since we last saw this specific person
            time_since_last = now - last_seen_times.get(identity, 0)
            
            # If it's a new ID interrupting, or an old ID returning after cooldown
            if identity != display_identity:
                if time_since_last > COOLDOWN_TIME:
                    print(f"Starting display for: {identity}")
                    display_identity = identity
                    current_images = load_images_for(display_identity)
                    current_index = 0
                    last_slide_time = now # Reset timer so the first slide gets full duration
            
            # Always update the timestamp while the person remains in frame
            last_seen_times[identity] = now

    # -------- Slideshow --------
    if now - last_slide_time > SLIDE_DURATION:
        last_slide_time = now
        prev = resize_fit(current_images[current_index])
        
        current_index += 1
        
        # Check if we played all images for the recognized ID
        if display_identity != "STOCK" and current_index >= len(current_images):
            print("Finished sequence. Returning to STOCK.")
            display_identity = "STOCK"
            current_images = load_images_for("STOCK")
            current_index = 0
        else:
            # Otherwise, just wrap around (standard behavior for STOCK)
            current_index = current_index % len(current_images)
            
        next_img = resize_fit(current_images[current_index])
        fade(prev, next_img)

    display_img = resize_fit(current_images[current_index])
    label = f"ID: {display_identity}"
    display_img = overlay_id(display_img, label)

    cv2.imshow(WINDOW_NAME, display_img)

    key = cv2.waitKey(1) & 0xFF

    if key in (27, ord('q')):
        break

    if key == ord('f'):
        fullscreen = not fullscreen
        mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)

cap.release()
cv2.destroyAllWindows()
