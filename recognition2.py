import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from scrfd import SCRFD

# ---------------- CONFIG ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

DB_DIR = "DB/embeddings"

MIN_FACE_SIZE = 80
MATCH_THRESHOLD = 1.2
PROCESS_INTERVAL = 1.5
BURST_FRAMES = 5
PRECHECK_DOWNSCALE = 0.5
RECOGNITION_COOLDOWN = 3.0

assert os.path.exists(SCRFD_MODEL), "SCRFD missing"
assert os.path.exists(ARCFACE_MODEL), "ArcFace missing"
assert os.path.exists(DB_DIR), "DB missing"

# ---------------- LOAD MODELS ----------------
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

# ---------------- LOAD DB ----------------
print("Loading DB...")

db_embeddings = []
db_ids = []

for file in os.listdir(DB_DIR):
    if file.endswith(".npy"):
        emb = np.load(os.path.join(DB_DIR, file))
        db_embeddings.append(emb)
        db_ids.append(os.path.splitext(file)[0])

db_embeddings = np.array(db_embeddings)

print(f"Loaded {len(db_ids)} identities")

# ---------------- CAMERA ----------------
pipeline = (
    "libcamerasrc ! "
    "video/x-raw,width=1640,height=1232,framerate=30/1 ! "
    "videoconvert ! appsink"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera open failed")
    exit(1)

print("\nAuto recognition running...\n")

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
    best_distance = float(distances[idx])

    if best_distance < MATCH_THRESHOLD:
        return db_ids[idx], best_distance
    else:
        return "UNKNOWN", best_distance

# ---------------- LOOP ----------------
last_process_time = 0
last_recognition_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    now = time.time()

    # UI Preview (optional)
    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

    if now - last_process_time < PROCESS_INTERVAL:
        continue

    last_process_time = now

    # ---------------- PRECHECK ----------------
    small = cv2.resize(frame, (0, 0),
                       fx=PRECHECK_DOWNSCALE,
                       fy=PRECHECK_DOWNSCALE)

    bboxes, _ = detector.detect(small, thresh=0.5, input_size=(320, 320))

    if bboxes is None or len(bboxes) == 0:
        continue  # silent wait

    print("\nFace detected → Burst capture")

    # ---------------- BURST ----------------
    faces = []

    for _ in range(BURST_FRAMES):
        ret, burst_frame = cap.read()
        if not ret:
            continue

        bboxes, _ = detector.detect(burst_frame, thresh=0.3, input_size=(640, 640))
        if bboxes is None:
            continue

        for bbox in bboxes:
            coords = valid_face(bbox, burst_frame.shape)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            score = float(bbox[4])
            area = (x2 - x1) * (y2 - y1)
            weighted = score * area

            face_crop = burst_frame[y1:y2, x1:x2]

            faces.append((weighted, face_crop))

    if not faces:
        print("No valid face in burst")
        continue

    _, best_face = max(faces, key=lambda x: x[0])

    if now - last_recognition_time < RECOGNITION_COOLDOWN:
        continue

    last_recognition_time = now

    # ---------------- EMBEDDING ----------------
    try:
        embedding = get_embedding(best_face)
    except Exception as e:
        print("Embedding error:", e)
        continue

    identity, distance = match_embedding(embedding)

    # ---------------- OUTPUT ----------------
    print("\nResult:")
    print("ID       :", identity)
    print("Distance :", round(distance, 4))

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
print("\nShutdown")
