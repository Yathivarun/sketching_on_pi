"""
Pi Display Module
Manages display window for showing:
1. Random slideshow when idle
2. Matched person images when recognition hit

OPTIMIZED FOR RASPBERRY PI:
- Efficient image loading and caching
- Low memory footprint
- Smooth transitions
"""

import cv2
import numpy as np
import time
import threading
from pathlib import Path
from typing import List, Optional
import random

from pi_config import *

# ============================================================================
# DISPLAY MANAGER
# ============================================================================

class PiDisplayManager:
    """
    Manages the display window on Raspberry Pi.
    
    Two modes:
    1. SLIDESHOW: Random images from stock_images/ folder
    2. MATCHED: Display specific person's images from laptop
    """
    
    def __init__(self):
        self.running = False
        self.window_name = "Pi Display"
        
        # Display state
        self.mode = "slideshow"  # "slideshow" or "matched"
        self.current_images = []
        self.current_index = 0
        self.last_switch_time = time.time()
        
        # Slideshow cache
        self.stock_images = []
        self.stock_image_paths = []
        
        # Matched person data
        self.matched_person_id = None
        self.matched_score = 0.0
        self.matched_images_data = []
        self.match_cycles_remaining = 0
        
        # Threading
        self.display_thread = None
        self.lock = threading.Lock()
        
        # Performance
        self.preload_next = None  # Preload next image for smooth transitions
    
    def load_stock_images(self):
        """Load all stock images for slideshow."""
        print("[DISPLAY] Loading stock images...")
        
        if not STOCK_IMAGES_DIR.exists():
            print(f"[DISPLAY] ⚠️ Stock images directory not found: {STOCK_IMAGES_DIR}")
            print(f"[DISPLAY] Creating directory. Please add images to: {STOCK_IMAGES_DIR}")
            STOCK_IMAGES_DIR.mkdir(exist_ok=True)
            return
        
        # Find all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(STOCK_IMAGES_DIR.glob(ext))
            image_paths.extend(STOCK_IMAGES_DIR.glob(ext.upper()))
        
        if not image_paths:
            print(f"[DISPLAY] ⚠️ No stock images found in {STOCK_IMAGES_DIR}")
            print("[DISPLAY] Please add images to display during idle time")
            return
        
        # Limit to avoid memory issues
        if len(image_paths) > MAX_CACHED_IMAGES:
            print(f"[DISPLAY] Found {len(image_paths)} images, limiting to {MAX_CACHED_IMAGES}")
            random.shuffle(image_paths)
            image_paths = image_paths[:MAX_CACHED_IMAGES]
        
        self.stock_image_paths = image_paths
        print(f"[DISPLAY] ✓ Loaded {len(self.stock_image_paths)} stock images")
        
        # Shuffle for variety
        random.shuffle(self.stock_image_paths)
    
    def start(self):
        """Start display window."""
        print("[DISPLAY] Starting display window...")
        
        # Load stock images
        self.load_stock_images()
        
        if not self.stock_image_paths:
            print("[DISPLAY] ⚠️ No images to display. Starting anyway...")
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        if DISPLAY_FULLSCREEN:
            cv2.setWindowProperty(
                self.window_name,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )
        else:
            cv2.resizeWindow(self.window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        # Start in slideshow mode
        self.mode = "slideshow"
        self.running = True
        
        # Start display thread
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        
        print("[DISPLAY] ✓ Display started")
    
    def trigger_match(self, person_id: str, score: float, images_data: List[bytes]):
        """
        Trigger matched person display.
        
        Args:
            person_id: Person ID (e.g., "1001")
            score: Match confidence score
            images_data: List of JPEG image data (bytes)
        """
        with self.lock:
            print(f"\n[DISPLAY] 🎯 MATCH TRIGGERED: Person #{person_id} (score: {score:.3f})")
            print(f"[DISPLAY] Received {len(images_data)} images")
            
            # Save images to temp directory
            RECEIVED_IMAGES_DIR.mkdir(exist_ok=True)
            
            # Clear old images
            for old_img in RECEIVED_IMAGES_DIR.glob("*.jpg"):
                old_img.unlink()
            
            # Save new images
            saved_paths = []
            for i, img_data in enumerate(images_data):
                img_path = RECEIVED_IMAGES_DIR / f"{person_id}_scene_{i+1}.jpg"
                with open(img_path, 'wb') as f:
                    f.write(img_data)
                saved_paths.append(img_path)
            
            # Update state
            self.matched_person_id = person_id
            self.matched_score = score
            self.matched_images_data = images_data
            self.current_images = saved_paths
            self.current_index = 0
            self.match_cycles_remaining = MATCH_DISPLAY_CYCLES
            self.mode = "matched"
            self.last_switch_time = time.time()
            
            print(f"[DISPLAY] → Switched to MATCHED mode")
    
    def return_to_slideshow(self):
        """Return to slideshow mode."""
        with self.lock:
            if self.mode != "slideshow":
                print("[DISPLAY] → Returning to SLIDESHOW mode")
                self.mode = "slideshow"
                self.matched_person_id = None
                self.matched_score = 0.0
                self.matched_images_data = []
                self.current_index = 0
                self.last_switch_time = time.time()
    
    def _display_loop(self):
        """Main display loop (runs in background thread)."""
        print("[DISPLAY] Display loop started")
        
        while self.running:
            try:
                with self.lock:
                    mode = self.mode
                
                if mode == "slideshow":
                    self._display_slideshow_frame()
                elif mode == "matched":
                    self._display_matched_frame()
                
                # Small delay to prevent CPU hogging
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"[DISPLAY] Error in display loop: {e}")
                time.sleep(1)
        
        print("[DISPLAY] Display loop stopped")
    
    def _display_slideshow_frame(self):
        """Display one frame of slideshow."""
        if not self.stock_image_paths:
            # No images, show blank screen with message
            blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            cv2.putText(
                blank,
                "Waiting for face detection...",
                (DISPLAY_WIDTH//2 - 300, DISPLAY_HEIGHT//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3
            )
            cv2.imshow(self.window_name, blank)
            cv2.waitKey(1)
            return
        
        # Check if it's time to switch image
        current_time = time.time()
        elapsed = current_time - self.last_switch_time
        
        if elapsed >= SLIDESHOW_INTERVAL:
            # Move to next image
            self.current_index = (self.current_index + 1) % len(self.stock_image_paths)
            self.last_switch_time = current_time
            
            # Reshuffle when we complete a cycle
            if self.current_index == 0:
                random.shuffle(self.stock_image_paths)
        
        # Load and display current image
        img_path = self.stock_image_paths[self.current_index]
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"[DISPLAY] ⚠️ Failed to load: {img_path}")
            return
        
        # Resize to fit display
        img_resized = self._resize_to_fit(img, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        cv2.imshow(self.window_name, img_resized)
        cv2.waitKey(1)
    
    def _display_matched_frame(self):
        """Display one frame of matched person images."""
        if not self.current_images:
            # No images, return to slideshow
            self.return_to_slideshow()
            return
        
        # Check if it's time to switch image
        current_time = time.time()
        elapsed = current_time - self.last_switch_time
        
        if elapsed >= MATCH_DISPLAY_INTERVAL:
            # Move to next image
            self.current_index += 1
            
            # Check if we completed a cycle
            if self.current_index >= len(self.current_images):
                self.current_index = 0
                self.match_cycles_remaining -= 1
                
                # Check if we should return to slideshow
                if self.match_cycles_remaining <= 0:
                    self.return_to_slideshow()
                    return
            
            self.last_switch_time = current_time
        
        # Load and display current image
        img_path = self.current_images[self.current_index]
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"[DISPLAY] ⚠️ Failed to load: {img_path}")
            return
        
        # Resize to fit display
        img_resized = self._resize_to_fit(img, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        
        # Add overlay with person info
        img_with_overlay = self._add_match_overlay(
            img_resized,
            self.matched_person_id,
            self.matched_score,
            self.current_index + 1,
            len(self.current_images)
        )
        
        cv2.imshow(self.window_name, img_with_overlay)
        cv2.waitKey(1)
    
    def _resize_to_fit(self, img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        Resize image to fit display while maintaining aspect ratio.
        Adds black borders if needed.
        """
        h, w = img.shape[:2]
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Resize
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create canvas
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # Center image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        return canvas
    
    def _add_match_overlay(
        self,
        img: np.ndarray,
        person_id: str,
        score: float,
        current_img: int,
        total_imgs: int
    ) -> np.ndarray:
        """Add informational overlay to matched image."""
        img_copy = img.copy()
        
        # Semi-transparent overlay bar at top
        overlay = img_copy.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, img_copy, 0.4, 0, img_copy)
        
        # Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Person ID
        cv2.putText(
            img_copy,
            f"Person #{person_id}",
            (20, 35),
            font,
            1.0,
            (0, 255, 0),
            2
        )
        
        # Confidence score
        cv2.putText(
            img_copy,
            f"Confidence: {score:.1%}",
            (20, 65),
            font,
            0.7,
            (255, 255, 255),
            2
        )
        
        # Image counter (right side)
        counter_text = f"{current_img}/{total_imgs}"
        text_size = cv2.getTextSize(counter_text, font, 0.8, 2)[0]
        cv2.putText(
            img_copy,
            counter_text,
            (img.shape[1] - text_size[0] - 20, 45),
            font,
            0.8,
            (255, 255, 255),
            2
        )
        
        return img_copy
    
    def stop(self):
        """Stop display."""
        self.running = False
        
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        cv2.destroyWindow(self.window_name)
        print("[DISPLAY] Stopped")


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

# This will be imported by main Pi application
pi_display = PiDisplayManager()


# ============================================================================
# STANDALONE TEST
# ============================================================================

def test_display():
    """Test display standalone."""
    import time
    
    print("Testing Pi Display Manager...")
    
    display = PiDisplayManager()
    display.start()
    
    try:
        # Run slideshow for 10 seconds
        print("Running slideshow for 10 seconds...")
        time.sleep(10)
        
        # Simulate a match
        print("\nSimulating match...")
        
        # Create dummy image
        dummy_img = np.random.randint(0, 255, (800, 600, 3), dtype=np.uint8)
        _, encoded = cv2.imencode('.jpg', dummy_img)
        dummy_data = encoded.tobytes()
        
        display.trigger_match(
            person_id="TEST123",
            score=0.95,
            images_data=[dummy_data, dummy_data, dummy_data]
        )
        
        # Wait for match display
        print("Displaying match for 30 seconds...")
        time.sleep(30)
        
        # Should auto-return to slideshow
        print("Should return to slideshow now...")
        time.sleep(10)
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        display.stop()


if __name__ == "__main__":
    test_display()
