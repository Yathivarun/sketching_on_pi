"""
Display Client for Face Recognition System
Displays stock images and switches to visitor-specific images when IDs are received.
Can run standalone or integrated with face_pi2.py.
"""

import cv2
import numpy as np
import os
import time
import threading
import queue
from collections import deque
from pathlib import Path
import requests
from typing import Optional, List, Dict

# Configuration
STOCK_IMAGES_DIR = "stock_images"
DISPLAY_MODE = "normal"  # Options: "normal" or "queue"
QUEUE_SIZE = 5
IMAGE_DISPLAY_DURATION = 5  # seconds
FADE_DURATION = 0.5  # seconds for fade effect
MINIO_SERVER_URL = "http://192.168.1.100:9000"  # Dummy MinIO endpoint
WINDOW_NAME = "Visitor Display"


class ImageCache:
    """Cache for downloaded visitor images"""
    def __init__(self):
        self.cache = {}  # visitor_id -> list of image arrays
        self.lock = threading.Lock()
    
    def has_images(self, visitor_id: str) -> bool:
        with self.lock:
            return visitor_id in self.cache and len(self.cache[visitor_id]) > 0
    
    def get_images(self, visitor_id: str) -> Optional[List[np.ndarray]]:
        with self.lock:
            return self.cache.get(visitor_id)
    
    def set_images(self, visitor_id: str, images: List[np.ndarray]):
        with self.lock:
            self.cache[visitor_id] = images
    
    def clear(self, visitor_id: str):
        with self.lock:
            if visitor_id in self.cache:
                del self.cache[visitor_id]


class DisplayClient:
    def __init__(self, mode: str = "normal"):
        self.mode = mode
        self.stock_images = []
        self.current_stock_idx = 0
        self.stock_switch_time = time.time()
        
        # Queue for visitor IDs
        self.visitor_queue = deque(maxlen=QUEUE_SIZE)
        self.queue_lock = threading.Lock()
        
        # Currently displaying visitor
        self.current_visitor_id = None
        self.display_start_time = None
        self.is_displaying_visitor = False
        
        # Set to track visitors in queue or being displayed
        self.active_visitors = set()
        
        # Image cache
        self.image_cache = ImageCache()
        
        # Communication queue for receiving visitor IDs
        self.id_queue = queue.Queue()
        
        # Window size (will be set based on first image)
        self.window_width = 1920
        self.window_height = 1080
        
        # Running flag
        self.running = True
        
        # Load stock images
        self._load_stock_images()
        
        # Create window (FIX 1: Fullscreen)
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
    def _load_stock_images(self):
        """Load stock images from directory"""
        stock_dir = Path(STOCK_IMAGES_DIR)
        
        if not stock_dir.exists():
            print(f"Warning: {STOCK_IMAGES_DIR} not found. Creating dummy stock images.")
            stock_dir.mkdir(parents=True, exist_ok=True)
            self._create_dummy_stock_images(stock_dir)
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = sorted([
            f for f in stock_dir.iterdir() 
            if f.suffix.lower() in image_extensions
        ])
        
        if not image_files:
            print("No stock images found. Creating dummy images.")
            self._create_dummy_stock_images(stock_dir)
            image_files = sorted([
                f for f in stock_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ])
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                self.stock_images.append(img)
                print(f"Loaded stock image: {img_path.name}")
        
        if not self.stock_images:
            print("Error: No valid stock images found!")
            self.stock_images.append(np.zeros((1080, 1920, 3), dtype=np.uint8))
    
    def _create_dummy_stock_images(self, stock_dir: Path):
        """Create dummy stock images for testing"""
        colors = [
            (100, 100, 200),
            (100, 200, 100),
            (200, 100, 100),
        ]
        
        for i, color in enumerate(colors):
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            img[:] = color
            
            text = f"Stock Image {i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 3, 4)[0]
            text_x = (1920 - text_size[0]) // 2
            text_y = (1080 + text_size[1]) // 2
            cv2.putText(img, text, (text_x, text_y), font, 3, (255, 255, 255), 4)
            
            img_path = stock_dir / f"stock_{i+1}.png"
            cv2.imwrite(str(img_path), img)
    
    def _fetch_images_from_minio(self, visitor_id: str) -> List[np.ndarray]:
        print(f"Fetching images for visitor ID: {visitor_id}")
        
        try:
            images = []
            num_images = 3
            
            for i in range(num_images):
                img = np.zeros((1080, 1920, 3), dtype=np.uint8)
                colors = [(50, 200, 50), (200, 50, 200), (200, 200, 50)]
                img[:] = colors[i % len(colors)]
                
                text1 = f"Visitor ID: {visitor_id}"
                text2 = f"Image {i+1}/{num_images}"
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, text1, (600, 400), font, 2, (255, 255, 255), 3)
                cv2.putText(img, text2, (700, 500), font, 1.5, (255, 255, 255), 3)
                
                images.append(img)
            
            print(f"Fetched {len(images)} images for visitor {visitor_id}")
            return images
            
        except Exception as e:
            print(f"Error fetching images for {visitor_id}: {e}")
            return []
    
    def _resize_image_to_fit(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        
        scale_w = self.window_width / w
        scale_h = self.window_height / h
        scale = min(scale_w, scale_h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        
        y_offset = (self.window_height - new_h) // 2
        x_offset = (self.window_width - new_w) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        return canvas
    
    # FIX 2: Smooth Fade Transition
    def _fade_transition(self, img1: np.ndarray, img2: np.ndarray) -> None:
        """Smooth fade from img1 to img2"""
        
        fps = 60
        steps = max(1, int(FADE_DURATION * fps))
        delay = int(1000 / fps)

        for i in range(steps + 1):
            alpha = i / steps
            blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            cv2.imshow(WINDOW_NAME, blended)

            if cv2.waitKey(delay) & 0xFF == 27:
                break
    
    def receive_visitor_id(self, visitor_id: str):
        self.id_queue.put(visitor_id)
    
    def _process_visitor_queue(self):
        try:
            while not self.id_queue.empty():
                visitor_id = self.id_queue.get_nowait()
                
                if visitor_id in self.active_visitors:
                    print(f"Visitor {visitor_id} already active, skipping")
                    continue
                
                if not self.image_cache.has_images(visitor_id):
                    images = self._fetch_images_from_minio(visitor_id)
                    if images:
                        self.image_cache.set_images(visitor_id, images)
                    else:
                        print(f"No images fetched for {visitor_id}, skipping")
                        continue
                
                self.active_visitors.add(visitor_id)
                
                if self.mode == "normal":
                    with self.queue_lock:
                        self.visitor_queue.clear()
                        self.visitor_queue.append(visitor_id)
                    print(f"NORMAL mode: Switching to visitor {visitor_id}")
                
                elif self.mode == "queue":
                    with self.queue_lock:
                        self.visitor_queue.append(visitor_id)
                    print(f"QUEUE mode: Added visitor {visitor_id} to queue")
        
        except queue.Empty:
            pass
    
    def _get_next_image(self) -> np.ndarray:
        current_time = time.time()
        
        if self.is_displaying_visitor and self.current_visitor_id:
            if current_time - self.display_start_time >= IMAGE_DISPLAY_DURATION:
                visitor_images = self.image_cache.get_images(self.current_visitor_id)
                
                if visitor_images and hasattr(self, 'current_visitor_img_idx'):
                    self.current_visitor_img_idx += 1
                    
                    if self.current_visitor_img_idx < len(visitor_images):
                        self.display_start_time = current_time
                        return self._resize_image_to_fit(visitor_images[self.current_visitor_img_idx])
                
                print(f"Finished displaying visitor {self.current_visitor_id}")
                self.active_visitors.discard(self.current_visitor_id)
                self.current_visitor_id = None
                self.is_displaying_visitor = False
                
                return self._get_stock_image()
            else:
                visitor_images = self.image_cache.get_images(self.current_visitor_id)
                if visitor_images and hasattr(self, 'current_visitor_img_idx'):
                    return self._resize_image_to_fit(visitor_images[self.current_visitor_img_idx])
        
        with self.queue_lock:
            if self.visitor_queue and not self.is_displaying_visitor:
                visitor_id = self.visitor_queue.popleft()
                visitor_images = self.image_cache.get_images(visitor_id)
                
                if visitor_images:
                    print(f"Starting to display visitor {visitor_id}")
                    self.current_visitor_id = visitor_id
                    self.current_visitor_img_idx = 0
                    self.is_displaying_visitor = True
                    self.display_start_time = current_time
                    return self._resize_image_to_fit(visitor_images[0])
        
        return self._get_stock_image()
    
    def _get_stock_image(self) -> np.ndarray:
        current_time = time.time()
        
        if current_time - self.stock_switch_time >= IMAGE_DISPLAY_DURATION:
            self.current_stock_idx = (self.current_stock_idx + 1) % len(self.stock_images)
            self.stock_switch_time = current_time
        
        return self._resize_image_to_fit(self.stock_images[self.current_stock_idx])
    
    def run(self):
        print(f"Display client starting in {self.mode.upper()} mode")
        print(f"Window: {self.window_width}x{self.window_height}")
        
        current_img = self._get_next_image()
        cv2.imshow(WINDOW_NAME, current_img)
        
        while self.running:
            self._process_visitor_queue()
            next_img = self._get_next_image()
            
            if not np.array_equal(current_img, next_img):
                self._fade_transition(current_img, next_img)
                current_img = next_img
            else:
                cv2.imshow(WINDOW_NAME, current_img)
            
            key = cv2.waitKey(100) & 0xFF
            if key == 27 or key == ord('q'):
                break
        
        self.shutdown()
    
    def shutdown(self):
        print("Shutting down display client...")
        self.running = False
        cv2.destroyAllWindows()


_display_client_instance = None
_display_thread = None


def start_display_client(mode: str = DISPLAY_MODE):
    global _display_client_instance, _display_thread
    
    if _display_client_instance is not None:
        print("Display client already running")
        return _display_client_instance
    
    _display_client_instance = DisplayClient(mode=mode)
    _display_thread = threading.Thread(target=_display_client_instance.run, daemon=True)
    _display_thread.start()
    
    print(f"Display client started in separate thread (mode: {mode})")
    return _display_client_instance


def send_visitor_id(visitor_id: str):
    global _display_client_instance
    
    if _display_client_instance is None:
        print("Warning: Display client not started")
        return
    
    _display_client_instance.receive_visitor_id(visitor_id)
    print(f"Sent visitor ID {visitor_id} to display client")


if __name__ == "__main__":
    print("=" * 60)
    print("Display Client - Standalone Mode")
    print("=" * 60)
    print(f"Mode: {DISPLAY_MODE}")
    print("Press ESC or Q to quit")
    print("=" * 60)
    
    client = DisplayClient(mode=DISPLAY_MODE)
    
    try:
        client.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        client.shutdown()
