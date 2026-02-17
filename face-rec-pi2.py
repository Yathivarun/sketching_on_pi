import cv2
import numpy as np
import onnxruntime as ort
import requests
import os
import time
from scrfd import SCRFD

SERVER_URL = "http://172.16.0.131:8000"

HEADLESS = True
PROCESS_INTERVAL = 2.0
BURST_FRAMES = 5
MIN_FACE_SIZE = 80
NN_COUNT = 1
CONF_THRESHOLD = 0.6
RECOGNITION_COOLDOWN = 5.0
PRECHECK_DOWNSCALE = 0.5

BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD_MODEL = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(BASE, "glintr100.onnx")

assert os.path.exists(SCRFD_MODEL), "SCRFD model missing"
assert os.path.exists(ARCFACE_MODEL), "ArcFace model missing"

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

def open_camera():
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("Camera open failed")
        return None
    print("Camera opened")
    return cap

cap = open_camera()
if cap is None:
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

def valid_face(bbox, frame_shape):
    x1, y1, x2, y2 = bbox[:4].astype(int)
    h, w = frame_shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    fw, fh = x2 - x1, y2 - y1
    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return None
    return (x1, y1, x2, y2)

def select_best_face(faces):
    return max(faces, key=lambda f: f["weighted"])

last_process_time = 0
last_success_time = 0
last_identity = None

print("\nRecognition running...\n")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed → reopening camera")
            cap.release()
            time.sleep(1)
            cap = open_camera()
            if cap is None:
                break
            continue
    except Exception as e:
        print("Camera crash:", e)
        cap.release()
        time.sleep(1)
        cap = open_camera()
        if cap is None:
            break
        continue

    if not HEADLESS:
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) == 27:
            break

    now = time.time()
    if now - last_process_time < PROCESS_INTERVAL:
        continue

    last_process_time = now

    small = cv2.resize(frame, (0, 0), fx=PRECHECK_DOWNSCALE, fy=PRECHECK_DOWNSCALE)
    bboxes, _ = detector.detect(small, thresh=0.5, input_size=(320, 320))

    if bboxes is None or len(bboxes) == 0:
        print("No face (precheck)")
        continue

    print("\n--- Burst ---")

    faces = []

    for _ in range(BURST_FRAMES):
        ret, burst_frame = cap.read()
        if not ret:
            continue

        bboxes, _ = detector.detect(burst_frame, thresh=0.3, input_size=(640, 640))
        if bboxes is None or len(bboxes) == 0:
            continue

        for bbox in bboxes:
            coords = valid_face(bbox, burst_frame.shape)
            if coords is None:
                continue

            x1, y1, x2, y2 = coords
            score = float(bbox[4])
            area = (x2 - x1) * (y2 - y1)
            weighted = score * area

            face = burst_frame[y1:y2, x1:x2]

            faces.append({
                "face": face,
                "score": score,
                "area": area,
                "weighted": weighted
            })

    if not faces:
        print("No valid face in burst")
        continue

    best = select_best_face(faces)
    print(f"Faces: {len(faces)} | Best score: {best['score']:.3f}")

    if now - last_success_time < RECOGNITION_COOLDOWN:
        print("Cooldown active")
        continue

    try:
        emb = get_embedding(best["face"])
    except Exception as e:
        print("Embedding error:", e)
        continue

    response = identify_face(emb)
    if not response:
        print("No server response")
        continue

    name = response.get("name", "UNKNOWN")
    confidence = float(response.get("confidence", 0))
    distance = float(response.get("distance", 0))
    visit_id = response.get("visit_id", "N/A")

    if confidence < CONF_THRESHOLD:
        name = "UNKNOWN"

    print("\nResult:")
    print("Name       :", name)
    print("Visit ID   :", visit_id)
    print("Confidence :", round(confidence, 3))
    print("Distance   :", round(distance, 3))

    if name != "UNKNOWN":
        last_success_time = now
        last_identity = name

cap.release()
cv2.destroyAllWindows()
print("\nShutdown")
