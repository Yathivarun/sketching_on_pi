#Local DB generation
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

MIN_FACE_SIZE = 80
PRECHECK_DOWNSCALE = 0.5

DB_DIR = "DB"
IMG_DIR = os.path.join(DB_DIR, "images")
EMB_DIR = os.path.join(DB_DIR, "embeddings")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

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
    print("Camera open failed")
    exit(1)

print("Camera opened")

# ---------------- INPUT ID ----------------
person_id = input("\nEnter ID for this capture: ").strip()

if not person_id:
    print("Invalid ID")
    exit(1)

img_path = os.path.join(IMG_DIR, f"{person_id}.jpg")
emb_path = os.path.join(EMB_DIR, f"{person_id}.npy")

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

# ---------------- LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    cv2.imshow("Enrollment Camera", frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC
        print("Exiting...")
        break

    if key == 32:  # SPACE
        print("\nCapturing...")

        small = cv2.resize(frame, (0, 0), fx=PRECHECK_DOWNSCALE, fy=PRECHECK_DOWNSCALE)
        bboxes, _ = detector.detect(small, thresh=0.5, input_size=(320, 320))

        if bboxes is None or len(bboxes) == 0:
            print("❌ No face detected. Try again.")
            continue

        bboxes, _ = detector.detect(frame, thresh=0.3, input_size=(640, 640))

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
            print("❌ Face too small / invalid.")
            continue

        _, (x1, y1, x2, y2) = max(faces, key=lambda x: x[0])

        face_crop = frame[y1:y2, x1:x2]

        try:
            embedding = get_embedding(face_crop)
        except Exception as e:
            print("Embedding error:", e)
            continue

        # Save image
        cv2.imwrite(img_path, frame)

        # Save embedding
        np.save(emb_path, embedding)

        print(f"✅ Saved:")
        print(f"Image → {img_path}")
        print(f"Embedding → {emb_path}")

        break

# ---------------- CLEANUP ----------------
cap.release()
cv2.destroyAllWindows()
print("\nDone.")
