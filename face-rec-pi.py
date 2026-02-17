import cv2
import numpy as np
import onnxruntime as ort
import requests
import os
import time
from scrfd import SCRFD

# ---------------- CONFIG ----------------
SERVER_URL = "http://172.16.0.131:8000"
PROCESS_INTERVAL = 2.0      # seconds between bursts
BURST_FRAMES = 5            # N frames per burst
MIN_FACE_SIZE = 80          # filter tiny faces
NN_COUNT = 1                # nearest neighbors

BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

assert os.path.exists(SCRFD_MODEL), "SCRFD model missing"
assert os.path.exists(ARCFACE_MODEL), "ArcFace model missing"

# ---------------- INIT MODELS ----------------
print("Loading models...")

detector = SCRFD(SCRFD_MODEL)
detector.prepare(-1)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

rec_sess = ort.InferenceSession(
    ARCFACE_MODEL, opts, providers=["CPUExecutionProvider"]
)
rec_input = rec_sess.get_inputs()[0].name

print("Models ready")

# ---------------- CAMERA ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Camera failed")
    exit(1)

print("Camera opened")

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
    img = preprocess_rec(face_bgr)
    emb = rec_sess.run(None, {rec_input: img})[0][0]
    emb = emb.astype(np.float32)
    return l2_normalize(emb)

def identify_face(embedding):
    try:
        vector_str = ",".join(map(str, embedding.tolist()))

        r = requests.post(
            f"{SERVER_URL}/api/v1/identify/",
            data={
                "type": "face",
                "face_vector": vector_str,
                "n": NN_COUNT
            },
            timeout=5
        )
        return r.json()

    except Exception as e:
        print("API error:", e)
        return None

def select_best_face(faces):
    return max(faces, key=lambda f: f["score"])

# ---------------- MAIN LOOP ----------------
print("\nRecognition started...\n")

last_process_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture failed")
        break

    now = time.time()
    if now - last_process_time < PROCESS_INTERVAL:
        continue

    last_process_time = now
    print("\n--- Burst Capture ---")

    faces = []

    for _ in range(BURST_FRAMES):
        ret, burst_frame = cap.read()
        if not ret:
            continue

        bboxes, kpss = detector.detect(
            burst_frame, thresh=0.3, input_size=(640, 640)
        )

        if bboxes is None or len(bboxes) == 0:
            continue

        for bbox in bboxes:
            bbox = bbox.astype(int)
            x1, y1, x2, y2 = bbox[:4]
            score = float(bbox[4])

            h, w = burst_frame.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            fw, fh = x2 - x1, y2 - y1
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue

            face = burst_frame[y1:y2, x1:x2]

            faces.append({
                "face": face,
                "score": score
            })

    if not faces:
        print("No valid face detected")
        continue

    best = select_best_face(faces)
    print(f"Faces detected: {len(faces)} | Using best score")

    try:
        emb = get_embedding(best["face"])
    except Exception as e:
        print("Embedding error:", e)
        continue

    response = identify_face(emb)

    if not response:
        print("No response from server")
        continue

    # ---------------- OUTPUT ----------------
    name = response.get("name", "UNKNOWN")
    confidence = response.get("confidence", 0)
    distance = response.get("distance", 0)
    visit_id = response.get("visit_id", "N/A")

    print("\nVisitor Result:")
    print("Name       :", name)
    print("Visit ID   :", visit_id)
    print("Confidence :", round(confidence, 3))
    print("Distance   :", round(distance, 3))

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
print("\nShutdown complete")
