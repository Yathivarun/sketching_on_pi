import cv2
import numpy as np
import onnxruntime as ort
import requests
import os
import time
from scrfd import SCRFD

# Import display client integration
try:
    from display_client import start_display_client, send_visitor_id
    DISPLAY_CLIENT_AVAILABLE = True
except ImportError:
    print("Warning: display_client.py not found. Running without display integration.")
    DISPLAY_CLIENT_AVAILABLE = False

BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

SERVER_URL = "http://192.168.1.100:8000"  # CHANGE THIS

assert os.path.exists(SCRFD_MODEL), "SCRFD model missing"
assert os.path.exists(ARCFACE_MODEL), "ArcFace model missing"

print("Models loaded:")
print("SCRFD :", SCRFD_MODEL)
print("ArcFace:", ARCFACE_MODEL)

# ---------------- Server Check ----------------
try:
    r = requests.get(f"{SERVER_URL}/health", timeout=3)
    print("Server reachable:", r.status_code)
except Exception as e:
    print("Server NOT reachable:", e)

# ---------------- SCRFD Detector ----------------
print("Initializing SCRFD...")
detector = SCRFD(SCRFD_MODEL)
detector.prepare(-1)
print("SCRFD ready")

# ---------------- ArcFace Session ----------------
print("Initializing ArcFace...")
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

rec_sess = ort.InferenceSession(
    ARCFACE_MODEL, opts, providers=["CPUExecutionProvider"]
)
rec_input = rec_sess.get_inputs()[0].name
print("ArcFace ready")

# ---------------- Display Client Integration ----------------
if DISPLAY_CLIENT_AVAILABLE:
    print("Starting display client...")
    display_client = start_display_client(mode="normal")  # Options: "normal" or "queue"
    print("Display client started")
    time.sleep(2)  # Give display client time to initialize
else:
    display_client = None

# ---------------- Camera ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)

print("Opening camera...")
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Camera failed")
    exit(1)

print("Camera opened")

# ---------------- Functions ----------------
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
    img = preprocess_rec(face_bgr)
    emb = rec_sess.run(None, {rec_input: img})[0][0]
    emb = emb.astype(np.float32)
    emb = l2_normalize(emb)
    print("Embedding generated | norm:", np.linalg.norm(emb))
    return emb

def identify(embedding):
    try:
        print("Sending embedding → server")
        r = requests.post(
            f"{SERVER_URL}/identify_by_face",
            json={"embedding": embedding.tolist()},
            timeout=5
        )
        print("Server response:", r.status_code)
        return r.json()
    except Exception as e:
        print("API error:", e)
        return {"status": "error", "message": str(e)}

# ---------------- Main Loop ----------------
print("Running recognition loop...\n")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    frame_id += 1
    print(f"\nFrame {frame_id}")

    try:
        bboxes, kpss = detector.detect(
            frame, thresh=0.3, input_size=(640, 640)
        )
    except Exception as e:
        print("Detection crash:", e)
        continue

    if bboxes is None or len(bboxes) == 0:
        print("No face detected")
        cv2.imshow("Pi Debug", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    print("Faces detected:", len(bboxes))

    bbox = bboxes[0].astype(int)
    x1, y1, x2, y2 = bbox[:4]

    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        print("Empty face crop")
        continue

    print("Face cropped:", face.shape)

    try:
        emb = get_embedding(face)
    except Exception as e:
        print("Embedding error:", e)
        continue

    response = identify(emb)

    if response.get("visitor"):
        name = response["visitor"]["name"]
        conf = response.get("confidence", 0)
        visitor_id = response["visitor"].get("id")  # Extract visitor ID
        
        label = f"{name} ({conf:.2f})"
        color = (0, 255, 0)
        print("Recognized:", name, "| confidence:", conf, "| ID:", visitor_id)
        
        # Send visitor ID to display client
        if DISPLAY_CLIENT_AVAILABLE and display_client and visitor_id:
            send_visitor_id(str(visitor_id))
            print(f"Visitor ID {visitor_id} sent to display client")
    else:
        conf = response.get("confidence", 0)
        label = f"UNKNOWN ({conf:.2f})"
        color = (0, 0, 255)
        print("Unknown face | confidence:", conf)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Pi Debug", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Shutdown complete")
