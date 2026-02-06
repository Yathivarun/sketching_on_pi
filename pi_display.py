#!/usr/bin/env python3
"""
Pi Display Module
Displays images on Pi's HDMI output or test window.
Matches the laptop's display.html behavior (4-second rotation, crossfade).
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from typing import List
import random


class PiDisplay:
    def __init__(self, fullscreen: bool = False, stock_dir: str = "stock_images"):
        self.fullscreen = fullscreen
        self.stock_dir = Path(stock_dir)

        self.current_mode = "slideshow"
        self.current_images = []
        self.current_index = 0

        self.stock_images = []
        self.stock_index = 0

        self.window_name = "Museum Display - Pi"
        self.running = False
        self.display_thread = None

        self.current_img = None
        self.next_img = None
        self.alpha = 0.0

        self.rotation_interval = 4.0
        self.crossfade_duration = 1.5
        self.last_switch_time = time.time()

        self.lock = threading.Lock()

        self.load_stock_images()

    # ------------------------------------------------------------------

    def load_stock_images(self):
        self.stock_images = []

        if not self.stock_dir.exists():
            self.stock_dir.mkdir(parents=True, exist_ok=True)

        for ext in ['*.jpg', '*.jpeg', '*.png']:
            for img_path in self.stock_dir.glob(ext):
                img = cv2.imread(str(img_path))
                if img is not None:
                    self.stock_images.append(img)

        if not self.stock_images:
            placeholder = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No Stock Images Found", (500, 540),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            self.stock_images = [placeholder]
        else:
            random.shuffle(self.stock_images)

        print(f"[DISPLAY] Loaded {len(self.stock_images)} stock images")

    # ------------------------------------------------------------------

    def start(self):
        if self.running:
            return

        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

    def stop(self):
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2.0)
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------

    def show_poi_images(self, image_data_list: List[bytes]):
        with self.lock:
            self.current_mode = "poi"
            self.current_images = []

            for img_bytes in image_data_list:
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    self.current_images.append(img)

            self.current_index = 0
            self.last_switch_time = time.time()

            # FIX: reset display buffers
            self.current_img = self._get_current_image()
            self.next_img = self._get_next_image()
            self.alpha = 0.0

        print(f"[DISPLAY] POI mode activated ({len(self.current_images)} images)")

    def return_to_slideshow(self):
        with self.lock:
            self.current_mode = "slideshow"
            self.current_images = []
            self.current_index = 0
            self.last_switch_time = time.time()

            # FIX: reset buffers
            self.current_img = self._get_current_image()
            self.next_img = self._get_next_image()
            self.alpha = 0.0

    # ------------------------------------------------------------------

    def _get_current_image(self):
        with self.lock:
            if self.current_mode == "poi" and self.current_images:
                return self.current_images[self.current_index].copy()
            return self.stock_images[self.stock_index].copy()

    def _get_next_image(self):
        with self.lock:
            if self.current_mode == "poi" and self.current_images:
                idx = (self.current_index + 1) % len(self.current_images)
                return self.current_images[idx].copy()
            idx = (self.stock_index + 1) % len(self.stock_images)
            return self.stock_images[idx].copy()

    def _advance_index(self):
        with self.lock:
            if self.current_mode == "poi" and self.current_images:
                self.current_index = (self.current_index + 1) % len(self.current_images)
            else:
                self.stock_index = (self.stock_index + 1) % len(self.stock_images)

    # ------------------------------------------------------------------

    def _resize_to_fit(self, img, target=(1920, 1080)):
        tw, th = target
        h, w = img.shape[:2]

        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)

        y = (th - nh) // 2
        x = (tw - nw) // 2
        canvas[y:y+nh, x:x+nw] = resized
        return canvas

    def _crossfade(self, img1, img2, alpha):
        return cv2.addWeighted(img1, 1.0 - alpha, img2, alpha, 0)

    # ------------------------------------------------------------------

    def _display_loop(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        if self.fullscreen:
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.current_img = self._get_current_image()
        self.next_img = self._get_next_image()

        while self.running:
            now = time.time()
            elapsed = now - self.last_switch_time

            if elapsed >= self.rotation_interval:
                fade_t = elapsed - self.rotation_interval

                if fade_t < self.crossfade_duration:
                    self.alpha = fade_t / self.crossfade_duration
                    frame = self._crossfade(self.current_img, self.next_img, self.alpha)
                else:
                    self._advance_index()
                    self.current_img = self.next_img
                    self.next_img = self._get_next_image()
                    self.alpha = 0.0
                    self.last_switch_time = now
                    frame = self.current_img
            else:
                frame = self.current_img

            if frame is None:
                continue

            frame = self._resize_to_fit(frame)
            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(33) == 27:
                self.running = False
                break

        cv2.destroyAllWindows()


# ----------------------------------------------------------------------

if __name__ == "__main__":
    display = PiDisplay(fullscreen=False, stock_dir="stock_images")
    display.start()
    time.sleep(10)
    display.stop()
