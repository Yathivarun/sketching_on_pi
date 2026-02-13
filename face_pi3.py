import cv2
import numpy as np
import onnxruntime as ort
import requests
import os
import time
from scrfd import SCRFD

BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

SERVER_URL = "http://192.168.1.100:8000"  # CHANGE

PROCESS_INTERVAL = 1.0
BURST_FRAMES = 5
MIN_FACE_SIZE = 100

assert os.path.exists(SCRFD_MODEL)
assert os.path.exists(ARCFACE_MODEL)

print("SCRFD :", SCRFD_MODEL)
print("ArcFace:", ARCFACE_MODEL)

try:
    r = requests.get(f"{SERVER_URL}/health", timeout=3)
    print("Server reachable:", r.status_code)
except Exception as e:
    print("Server NOT reachable:", e)

detector = SCRFD(SCRFD_MODEL)
detector.prepare(-1)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

rec_sess = ort.InferenceSession(
    ARCFACE_MODEL, opts, providers=["CPUExecutionProvider"]
)
rec_input = rec_sess.get_inputs()[0].name

pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera error")
    exit(1)

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

def identify(embedding):
    try:
        r = requests.post(
            f"{SERVER_URL}/identify_by_face",
            json={"embedding": embedding.tolist()},
            timeout=5
        )
        return r.json()
    except Exception as e:
        print("API error:", e)
        return None

def select_largest_face(bboxes):
    return max(
        bboxes,
        key=lambda b: (b[2] - b[0]) * (b[3] - b[1])
    )

last_process_time = 0

print("\nRecognition running...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame error")
        break

    now = time.time()
    if now - last_process_time < PROCESS_INTERVAL:
        cv2.imshow("Pi Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    last_process_time = now

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

        largest = select_largest_face(bboxes).astype(int)
        x1, y1, x2, y2 = largest[:4]

        h, w = burst_frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        fw, fh = x2 - x1, y2 - y1
        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            continue

        face = burst_frame[y1:y2, x1:x2]
        score = float(largest[4])

        faces.append({
            "face": face,
            "bbox": (x1, y1, x2, y2),
            "score": score
        })

    if not faces:
        print("No valid face in burst")
        cv2.imshow("Pi Recognition", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    best = max(faces, key=lambda f: f["score"])
    face = best["face"]
    x1, y1, x2, y2 = best["bbox"]

    try:
        emb = get_embedding(face)
    except Exception as e:
        print("Embedding error:", e)
        continue

    response = identify(emb)

    if response and response.get("visitor"):
        name = response["visitor"]["name"]
        conf = response.get("confidence", 0)
        label = f"{name} ({conf:.2f})"
        color = (0, 255, 0)
        print("Recognized:", name, "|", conf)
    else:
        conf = response.get("confidence", 0) if response else 0
        label = f"UNKNOWN ({conf:.2f})"
        color = (0, 0, 255)
        print("Unknown |", conf)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Pi Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Shutdown")
