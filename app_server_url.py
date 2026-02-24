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
from io import BytesIO
from scrfd import SCRFD

# Tkinter + Pillow for direct URL display (OpenCV cannot render URLs)
import tkinter as tk
from PIL import Image, ImageTk

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
FADE_STEPS         = 24       # frames for fade (matches ~0.4s at ~60fps feel)
COOLDOWN_TIME      = 20.0
REQUEST_TIMEOUT    = 5

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

# ---------------- TKINTER DISPLAY WINDOW ----------------
root = tk.Tk()
root.title("Display")
root.attributes("-fullscreen", True)
root.configure(bg="black")
root.geometry(f"{SCREEN_W}x{SCREEN_H}+0+0")

canvas = tk.Canvas(root, width=SCREEN_W, height=SCREEN_H,
                   bg="black", highlightthickness=0)
canvas.pack(fill=tk.BOTH, expand=True)

# Label for identity overlay (bottom-right corner, matches original position)
id_label = tk.Label(
    root, text="", font=("Helvetica", 18),
    fg="white", bg="black",
    padx=8, pady=4
)
id_label.place(
    x=int(SCREEN_W * 0.74),
    y=int(SCREEN_H * 0.93)
)

# Holds the current PhotoImage reference — must stay in scope or GC kills it
_current_photo = None
_prev_photo    = None

# ---------------- THREAD-SAFE QUEUES ----------------
image_queue   = queue.Queue(maxsize=1)   # background → main: new identity + PIL images
display_queue = queue.Queue(maxsize=10)  # main loop → tkinter thread: display commands

# ---------------- STATE ----------------
current_images   = []   # list of PIL.Image objects
current_index    = 0
last_slide_time  = 0
display_identity = "STOCK"
last_seen_times  = {}
bg_pending       = False

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

# ---------------- PIL IMAGE UTILS ----------------
def pil_resize_fit(pil_img, w=SCREEN_W, h=SCREEN_H):
    """Letterbox-fit a PIL image onto a black canvas of screen size."""
    iw, ih = pil_img.size
    scale   = min(w / iw, h / ih)
    new_w   = int(iw * scale)
    new_h   = int(ih * scale)
    resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
    canvas  = Image.new("RGB", (w, h), (0, 0, 0))
    x = (w - new_w) // 2
    y = (h - new_h) // 2
    canvas.paste(resized, (x, y))
    return canvas

def show_pil_image(pil_img):
    """Render a PIL image onto the tkinter canvas. Must be called from tkinter thread."""
    global _current_photo
    _current_photo = ImageTk.PhotoImage(pil_img)
    canvas.create_image(0, 0, anchor=tk.NW, image=_current_photo)
    canvas.update_idletasks()

def blend_pil(img1, img2, alpha):
    """Blend two same-size PIL images. alpha=0 → img1, alpha=1 → img2."""
    return Image.blend(img1, img2, alpha)

# ---------------- LOCAL STOCK LOADER ----------------
def load_stock_images():
    """Returns list of PIL.Image objects from local stock_images folder."""
    images = []
    for img_path in sorted(STOCK_DIR.glob("*")):
        try:
            img = Image.open(str(img_path)).convert("RGB")
            images.append(img)
        except Exception:
            pass
    if not images:
        images = [Image.new("RGB", (SCREEN_W, SCREEN_H), (0, 0, 0))]
    return images

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

def download_pil_from_url(url):
    """
    Download image from URL directly into a PIL.Image (no disk write).
    Returns PIL.Image or None on failure.
    """
    try:
        r   = requests.get(url, timeout=REQUEST_TIMEOUT)
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return img
    except Exception as e:
        print(f"Image URL load error ({url}):", e)
        return None

# ---------------- BACKGROUND WORKER ----------------
def background_identify_and_fetch(embedding):
    """
    Background thread:
    1. Identify via recognition server
    2. Fetch URLs from MinIO
    3. Download each URL into PIL.Image (in memory only)
    4. Push result to image_queue
    Falls back to stock images on any failure.
    """
    global bg_pending

    identity = identify_on_server(embedding)

    if identity == "UNKNOWN":
        bg_pending = False
        return

    urls = fetch_minio_urls(identity)

    if urls:
        images = [img for img in (download_pil_from_url(u) for u in urls) if img is not None]
    else:
        images = []

    if not images:
        print(f"No images for {identity}, falling back to stock")
        images   = load_stock_images()
        identity = "STOCK"

    try:
        image_queue.put_nowait({"identity": identity, "images": images})
    except queue.Full:
        pass  # Newer result already pending, discard this one

    bg_pending = False

# ---------------- FADE (PIL-based, same logic as original) ----------------
def do_fade(img1_pil, img2_pil):
    """Fade between two PIL images by blending and displaying each step."""
    img1_fit = pil_resize_fit(img1_pil)
    img2_fit = pil_resize_fit(img2_pil)
    for i in range(FADE_STEPS + 1):
        alpha   = i / FADE_STEPS
        blended = blend_pil(img1_fit, img2_fit, alpha)
        show_pil_image(blended)
        root.update()
        time.sleep(0.016)   # ~60fps

# ---------------- OVERLAY ID (tkinter label, same position as original) ----------------
def update_id_label(text):
    id_label.config(text=text)

# ---------------- KEYBOARD HANDLER ----------------
fullscreen = True

def on_key(event):
    global fullscreen
    if event.keysym == "Escape" or event.char == "q":
        shutdown()
    elif event.char == "f":
        fullscreen = not fullscreen
        root.attributes("-fullscreen", fullscreen)

root.bind("<Key>", on_key)

def shutdown():
    cap.release()
    root.destroy()

# ---------------- INIT ----------------
last_process_time = 0
current_images    = load_stock_images()
show_pil_image(pil_resize_fit(current_images[0]))
update_id_label("ID: STOCK")

# ---------------- MAIN LOOP (runs inside tkinter's event loop via after()) ----------------
def main_loop():
    global current_images, current_index, last_slide_time
    global display_identity, last_seen_times, bg_pending
    global last_process_time

    ret, frame = cap.read()
    if not ret:
        shutdown()
        return

    now = time.time()

    # -------- Consume queue from background thread --------
    try:
        result       = image_queue.get_nowait()
        new_identity = result["identity"]
        new_images   = result["images"]

        time_since_last = now - last_seen_times.get(new_identity, 0)

        if new_identity != display_identity and time_since_last > COOLDOWN_TIME:
            prev_pil         = pil_resize_fit(current_images[current_index])
            display_identity = new_identity
            current_images   = new_images
            current_index    = 0
            last_slide_time  = now
            do_fade(prev_pil, current_images[0])
            update_id_label(f"ID: {display_identity}")

    except queue.Empty:
        pass

    # -------- Face Detection + Embedding --------
    if now - last_process_time > PROCESS_INTERVAL and not bg_pending:
        last_process_time = now

        small = cv2.resize(frame, (0, 0),
                           fx=PRECHECK_DOWNSCALE,
                           fy=PRECHECK_DOWNSCALE)

        bboxes, _ = detector.detect(small, thresh=0.5, input_size=(320, 320))

        if bboxes is not None:
            faces = []

            for _ in range(BURST_FRAMES):
                ret2, burst = cap.read()
                if not ret2:
                    continue

                bboxes2, _ = detector.detect(burst, thresh=0.3, input_size=(640, 640))
                if bboxes2 is None:
                    continue

                for bbox in bboxes2:
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

                bg_pending = True
                t = threading.Thread(
                    target=background_identify_and_fetch,
                    args=(emb,),
                    daemon=True
                )
                t.start()

    # -------- Slideshow (same logic as original) --------
    if now - last_slide_time > SLIDE_DURATION:
        last_slide_time = now
        prev_pil = pil_resize_fit(current_images[current_index])

        current_index += 1

        if display_identity != "STOCK" and current_index >= len(current_images):
            display_identity = "STOCK"
            current_images   = load_stock_images()
            current_index    = 0
            update_id_label("ID: STOCK")
        else:
            current_index = current_index % len(current_images)

        next_pil = pil_resize_fit(current_images[current_index])
        do_fade(prev_pil, next_pil)

    # -------- Render current frame --------
    display_pil = pil_resize_fit(current_images[current_index])
    show_pil_image(display_pil)

    # Schedule next iteration (~16ms = ~60fps feel)
    root.after(16, main_loop)

# ---------------- START ----------------
time.sleep(2)
root.after(100, main_loop)
root.mainloop()
