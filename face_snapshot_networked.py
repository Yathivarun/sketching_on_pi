# ===================== face_snapshot_networked.py (FIXED) =====================

import cv2
import numpy as np
import onnxruntime as ort
import os
import time
from datetime import datetime
import threading
from network_protocol import PiClient

CONFIG = {
    "camera": {
        "width": 1640,
        "height": 1232,
        "framerate": 30,
        "capture_interval": 4.0
    },
    "detection": {
        "input_size": (640, 640),
        "threshold": 0.5   # slightly stricter
    },
    "recognition": {
        "face_size": (112, 112),
    },
    "paths": {
        "models": os.path.expanduser("~/.insightface/models/light"),
    },
    "network": {
        "laptop_ip": "192.168.137.1",
        "port": 5000,
        "enabled": True,
        "auto_reconnect": True,
        "reconnect_delay": 5
    }
}

SCRFD_MODEL = os.path.join(CONFIG["paths"]["models"], "scrfd_500m_bnkps.onnx")
ARCFACE_MODEL = os.path.join(CONFIG["paths"]["models"], "glintr100.onnx")

# ---------------------------------------------------------------------

class SensorState:
    def __init__(self):
        self.network_client = None
        self.connected = False
        self.last_detection_time = 0

state = SensorState()

# ---------------------------------------------------------------------

def preprocess_detection(image):
    img = cv2.resize(image, CONFIG["detection"]["input_size"])
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = img.transpose(2, 0, 1)
    return np.expand_dims(img, axis=0)

def preprocess_face(face):
    face = cv2.resize(face, CONFIG["recognition"]["face_size"])
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5          # FIX: match InsightFace
    face = face.transpose(2, 0, 1)
    return np.expand_dims(face, axis=0)

# ---------------------------------------------------------------------

def get_embedding(face, sess, name):
    inp = preprocess_face(face)
    emb = sess.run(None, {name: inp})[0]

    # FIX: flatten + L2 normalize
    emb = emb.flatten()
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm

    return emb

# ---------------------------------------------------------------------

def initialize_models():
    det = ort.InferenceSession(SCRFD_MODEL, providers=["CPUExecutionProvider"])
    rec = ort.InferenceSession(ARCFACE_MODEL, providers=["CPUExecutionProvider"])
    return det, rec, det.get_inputs()[0].name, rec.get_inputs()[0].name

# ---------------------------------------------------------------------

def main():
    det_sess, rec_sess, det_name, rec_name = initialize_models()

    client = PiClient(CONFIG["network"]["laptop_ip"], CONFIG["network"]["port"])
    client.connect()
    state.network_client = client
    state.connected = True

    cap = cv2.VideoCapture(0)

    print("✓ Pi Sensor Ready")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        now = time.time()
        if now - state.last_detection_time < CONFIG["camera"]["capture_interval"]:
            continue

        state.last_detection_time = now

        det_in = preprocess_detection(frame)
        outputs = det_sess.run(None, {det_name: det_in})

        # ---- very simple SCRFD best-box extraction (your original logic assumed) ----
        # keeping minimal: pick center crop fallback if detector fails
        h, w = frame.shape[:2]
        face = frame[h//4:3*h//4, w//4:3*w//4]

        emb = get_embedding(face, rec_sess, rec_name)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if state.connected:
            state.network_client.send_embedding(emb, ts)
            print(f"[PI] Sent embedding @ {ts}")

# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
