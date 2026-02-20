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
MATCH_THRESHOLD = 0.9   # adjust (lower = stricter)

assert os.path.exists(SCRFD_MODEL), "SCRFD model missing"
assert os.path.exists(ARCFACE_MODEL), "ArcFace model missing"
assert os.path.exists(DB_DIR), "DB embeddings folder missing"

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
print("Loading embeddings DB...")

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

print("\nPress SPACE to capture | ESC to exit\n")

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

def valid_face(bbox, shape):
    x1, y1, x2, y2 = bbox[:4].astype(int)
    h, w = shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    fw, fh = x2 - x1, y2 - y1
    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return None

    return (x1, y1, x2, y2)

def match_embedding(embedding):
    distances = np.linalg.norm(db_embeddings - embedding, axis=1)
    idx = np.argmin(distances)
    best_distance = distances[idx]

    if best_distance < MATCH_THRESHOLD:
        return db_ids[idx], best_distance
    else:
        return "UNKNOWN", best_distance

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("Recognition Camera", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("Exiting...")
        break

    if key == 32:  # SPACE
        print("\nCapturing...")

        bboxes, _ = detector.detect(frame, thresh=0.3, input_size=(640, 640))

        if bboxes is None or len(bboxes) == 0:
            print("❌ No face detected. Try again.")
            continue

        faces = []
        for bbox in bboxes:
            coords = valid_face(bbox, frame.shape)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            score = float(bbox[4])
            area = (x2 - x1) * (y2 - y1)
            weighted = score * area

            faces.append((weighted, (x1, y1, x2, y2)))

        if not faces:
            print("❌ Face too small.")
            continue

        _, (x1, y1, x2, y2) = max(faces, key=lambda x: x[0])

        face_crop = frame[y1:y2, x1:x2]

        try:
            embedding = get_embedding(face_crop)
        except Exception as e:
            print("Embedding error:", e)
            continue

        identity, distance = match_embedding(embedding)

        print("\nResult:")
        print("ID       :", identity)
        print("Distance :", round(float(distance), 4))

        print("\nReady for next capture...\n")

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
print("\nShutdown")
