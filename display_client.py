import cv2
import os
import time
import threading
from collections import deque

# ================= CONFIG =================
MODE = "latest"        # "latest" or "queue"
QUEUE_MAX = 5
SLIDE_TIME = 5         # seconds per image
FADE_TIME = 0.8        # fade duration
STOCK_DIR = "stock_images"
WINDOW_NAME = "Display"

# ============== STATE =====================
current_id = None
id_queue = deque()
running = True
lock = threading.Lock()

# ============== DUMMY MINIO FETCH =========
def fetch_images_from_minio(person_id):
    """
    Placeholder for MinIO fetch.
    Replace later with actual API call.
    """
    print(f"[MinIO] Fetching images for ID {person_id}")

    dummy_dir = f"person_images/{person_id}"
    if not os.path.exists(dummy_dir):
        print("[MinIO] No images found → fallback to stock")
        return []

    images = []
    for f in os.listdir(dummy_dir):
        path = os.path.join(dummy_dir, f)
        if path.lower().endswith((".png", ".jpg", ".jpeg")):
            images.append(path)

    print(f"[MinIO] {len(images)} images fetched")
    return images

# ============== IMAGE FIT =================
def fit_to_screen(img, screen_w, screen_h):
    h, w = img.shape[:2]
    scale = min(screen_w / w, screen_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    canvas = 255 * ( 
        (screen_h, screen_w, 3) 
    )
    canvas = canvas.astype("uint8")

    y_offset = (screen_h - new_h) // 2
    x_offset = (screen_w - new_w) // 2

    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

# ============== FADE ======================
def fade_transition(prev_img, next_img, screen_w, screen_h):
    steps = int(FADE_TIME * 30)
    for i in range(steps):
        alpha = i / steps
        blended = cv2.addWeighted(prev_img, 1-alpha, next_img, alpha, 0)
        cv2.imshow(WINDOW_NAME, blended)
        cv2.waitKey(30)

# ============== SLIDESHOW =================
def play_slideshow(images, fade=False):
    if not images:
        return

    screen_w = 1280
    screen_h = 720

    prev_frame = None

    for img_path in images:
        if not running:
            break

        img = cv2.imread(img_path)
        if img is None:
            continue

        frame = fit_to_screen(img, screen_w, screen_h)

        if fade and prev_frame is not None:
            fade_transition(prev_frame, frame, screen_w, screen_h)

        cv2.imshow(WINDOW_NAME, frame)

        start = time.time()
        while time.time() - start < SLIDE_TIME:
            if cv2.waitKey(30) == 27:
                exit(0)

        prev_frame = frame

# ============== STOCK LOOP ================
def stock_slideshow_loop():
    stock_images = [
        os.path.join(STOCK_DIR, f)
        for f in os.listdir(STOCK_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not stock_images:
        print("[Stock] No stock images found")
        return

    print("[Stock] Slideshow started")

    while running:
        with lock:
            if current_id is not None:
                time.sleep(0.1)
                continue

        play_slideshow(stock_images, fade=True)

# ============== ID LOOP ===================
def id_slideshow_loop():
    global current_id

    while running:
        with lock:
            if MODE == "queue":
                if current_id is None and id_queue:
                    current_id = id_queue.popleft()
            else:
                pass

        if current_id is None:
            time.sleep(0.1)
            continue

        images = fetch_images_from_minio(current_id)

        if images:
            print(f"[Display] Showing ID {current_id}")
            play_slideshow(images, fade=False)
        else:
            print("[Display] No images → fallback")

        with lock:
            print("[Display] Returning to stock")
            current_id = None

# ============== INPUT HANDLER =============
def receive_new_id(person_id):
    global current_id

    with lock:
        if person_id == current_id:
            print("[Logic] Duplicate ID → ignored")
            return

        if MODE == "latest":
            print(f"[Logic] Latest mode → switching to {person_id}")
            current_id = person_id
            id_queue.clear()

        elif MODE == "queue":
            if person_id not in id_queue:
                if len(id_queue) >= QUEUE_MAX:
                    removed = id_queue.popleft()
                    print(f"[Queue] Full → removed oldest {removed}")

                id_queue.append(person_id)
                print(f"[Queue] Added ID {person_id}")

# ============== SIMULATION (REMOVE LATER) ==
def simulate_server_input():
    test_ids = [1, 2, None, 2, 3]

    for pid in test_ids:
        time.sleep(8)

        if pid is None:
            print("[Sim] Unknown / No face")
            continue

        print(f"[Sim] New ID → {pid}")
        receive_new_id(pid)

# ============== MAIN ======================
def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    threading.Thread(target=stock_slideshow_loop, daemon=True).start()
    threading.Thread(target=id_slideshow_loop, daemon=True).start()

    simulate_server_input()  # REMOVE when integrated

    while running:
        if cv2.waitKey(30) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
