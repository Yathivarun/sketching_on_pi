import cv2
import numpy as np
import onnxruntime as ort
import os
from scrfd import SCRFD

# ---------------- CONFIG ----------------
BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

DB_DIR = "DB"
IMG_DIR = os.path.join(DB_DIR, "images")
EMB_DIR = os.path.join(DB_DIR, "embeddings")

MIN_FACE_SIZE = 80

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

assert os.path.exists(SCRFD_MODEL), "SCRFD model missing"
assert os.path.exists(ARCFACE_MODEL), "ArcFace model missing"

# ---------------- LOAD MODELS ----------------
print("Loading models...")

detector = SCRFD(SCRFD_MODEL)
detector.prepare(-1)

opts = ort.SessionOptions()
rec_sess = ort.InferenceSession(
    ARCFACE_MODEL, opts, providers=["CPUExecutionProvider"]
)
rec_input = rec_sess.get_inputs()[0].name

print("Models ready\n")

# ---------------- INPUTS ----------------
img_path = input("Enter image path: ").strip()
person_id = input("Enter person ID: ").strip()

if not os.path.exists(img_path):
    print("❌ Image path invalid")
    exit(1)

if not person_id:
    print("❌ Invalid ID")
    exit(1)

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

# ---------------- PROCESS IMAGE ----------------
img = cv2.imread(img_path)

if img is None:
    print("❌ Failed to load image")
    exit(1)

bboxes, _ = detector.detect(img, thresh=0.3, input_size=(640, 640))

if bboxes is None or len(bboxes) == 0:
    print("❌ No face detected")
    exit(1)

faces = []

for bbox in bboxes:
    coords = valid_face(bbox, img.shape)
    if coords is None:
        continue

    x1, y1, x2, y2 = coords
    score = float(bbox[4])
    area = (x2 - x1) * (y2 - y1)
    weighted = score * area

    faces.append((weighted, (x1, y1, x2, y2)))

if not faces:
    print("❌ Face too small / invalid")
    exit(1)

_, (x1, y1, x2, y2) = max(faces, key=lambda x: x[0])

face_crop = img[y1:y2, x1:x2]

try:
    embedding = get_embedding(face_crop)
except Exception as e:
    print("❌ Embedding error:", e)
    exit(1)

# ---------------- SAVE ----------------
save_img_path = os.path.join(IMG_DIR, f"{person_id}.jpg")
save_emb_path = os.path.join(EMB_DIR, f"{person_id}.npy")

cv2.imwrite(save_img_path, img)
np.save(save_emb_path, embedding)

print("\n✅ Enrollment successful")
print("Saved image →", save_img_path)
print("Saved embedding →", save_emb_path)
