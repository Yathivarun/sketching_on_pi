import cv2
import numpy as np
import onnxruntime as ort
import requests
import time
import os

BASE = os.path.expanduser("~/.insightface/models/light")
SCRFD = os.path.join(BASE, "scrfd_500m_bnkps.onnx")
ARCFACE = os.path.join(BASE, "glintr100.onnx")

SERVER_URL = "http://192.168.1.100:8000"  # change

assert os.path.exists(SCRFD)
assert os.path.exists(ARCFACE)

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.inter_op_num_threads = 2

det_sess = ort.InferenceSession(SCRFD, opts, providers=["CPUExecutionProvider"])
rec_sess = ort.InferenceSession(ARCFACE, opts, providers=["CPUExecutionProvider"])

det_input = det_sess.get_inputs()[0].name
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
    face = cv2.resize(face_bgr, (112, 112))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = (face.astype(np.float32) - 127.5) * 0.0078125
    face = face.transpose(2, 0, 1)
    return face[None, ...]

def l2_normalize(vec):
    norm = np.linalg.norm(vec)
    return vec if norm == 0 else vec / norm

def detect_face(frame):
    blob = cv2.dnn.blobFromImage(
        frame, 1.0 / 128, (640, 640),
        (127.5, 127.5, 127.5),
        swapRB=True
    )
    outputs = det_sess.run(None, {det_input: blob})
    scores = outputs[0].flatten()
    bboxes = outputs[1][0]

    if len(scores) == 0:
        return None

    idx = np.argmax(scores)
    if scores[idx] < 0.3:
        return None

    x1, y1, x2, y2 = bboxes[idx][:4]
    h, w = frame.shape[:2]

    x1 = int(max(0, min(x1, w - 1)))
    y1 = int(max(0, min(y1, h - 1)))
    x2 = int(max(x1 + 1, min(x2, w)))
    y2 = int(max(y1 + 1, min(y2, h)))

    if (x2 - x1) < 20 or (y2 - y1) < 20:
        return None

    return (x1, y1, x2, y2), float(scores[idx])

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
        return {"status": "error", "message": str(e)}

print("Running recognition...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detect_face(frame)

    if result:
        (x1, y1, x2, y2), score = result
        face = frame[y1:y2, x1:x2]

        if face.size > 0:
            emb = get_embedding(face)
            response = identify(emb)

            if response.get("visitor"):
                name = response["visitor"]["name"]
                conf = response.get("confidence", 0)
                label = f"{name} ({conf:.2f})"
                color = (0, 255, 0)
            else:
                conf = response.get("confidence", 0)
                label = f"UNKNOWN ({conf:.2f})"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow("Pi Face Recognition", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
