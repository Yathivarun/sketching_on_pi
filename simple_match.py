#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis
import time

# ------------------------------------------------------------------

EMBEDDINGS_DIR = Path("embeddings")
THRESHOLD = 0.5

# ------------------------------------------------------------------

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ------------------------------------------------------------------

print("[INFO] Loading InsightFace (antelopev2)...")

app = FaceAnalysis(
    name="antelopev2",
    root="./models/insightface",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=-1, det_size=(640, 640))

print("[INFO] Model ready")

# ------------------------------------------------------------------

# Load stored embeddings
known = {}

for f in EMBEDDINGS_DIR.glob("*.npy"):
    emb = np.load(f)
    emb = emb.flatten()
    emb = emb / np.linalg.norm(emb)
    known[f.stem] = emb

print(f"[INFO] Loaded {len(known)} stored embeddings")

# ------------------------------------------------------------------

cap = cv2.VideoCapture(0)

print("\nPress SPACE to capture")
print("Press ESC to exit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Pi Face Test", frame)
    key = cv2.waitKey(1)

    # ESC
    if key == 27:
        break

    # SPACE = capture
    if key == 32:
        print("\n[CAPTURE] Processing frame...")

        faces = app.get(frame)

        if not faces:
            print("❌ No face detected")
            continue

        # Largest face
        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        emb = face.embedding
        emb = emb.flatten()
        emb = emb / np.linalg.norm(emb)

        best_id = None
        best_score = 0

        for k, v in known.items():
            s = cosine(emb, v)
            if s > best_score:
                best_score = s
                best_id = k

        if best_score > THRESHOLD:
            print(f"✅ MATCH: {best_id}  score={best_score:.3f}")
        else:
            print(f"❌ NO MATCH (best={best_score:.3f})")

        time.sleep(0.5)

cap.release()
cv2.destroyAllWindows()
