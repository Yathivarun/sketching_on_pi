"""
Pi Main Application
Integrates face detection and display modules.

This is the main entry point for the Raspberry Pi.
Run this file to start the Pi sensor system.

Workflow:
1. Start display in slideshow mode
2. Start face detection camera
3. On capture: detect face → send embedding → wait for match
4. On match: trigger display to show person images
5. After timeout: return to slideshow
"""

import sys
import time
import threading
from pathlib import Path

# Import Pi modules
from pi_config import *
from pi_face_detect import PiFaceCapture
from pi_display import PiDisplayManager

# ============================================================================
# MAIN PI APPLICATION
# ============================================================================

class PiApplication:
    """
    Main Pi application coordinator.
    Manages both face detection and display modules.
    """
    
    def __init__(self):
        self.running = False
        
        # Modules
        self.face_capture = None
        self.display_manager = None
        
        # State
        self.waiting_for_match = False
    
    def initialize(self):
        """Initialize all modules."""
        print("\n" + "="*60)
        print("🔷 INITIALIZING PI SYSTEM")
        print("="*60)
        
        # 1. Initialize Display
        print("\n[1/3] Starting Display Manager...")
        self.display_manager = PiDisplayManager()
        self.display_manager.start()
        time.sleep(1)  # Give display time to initialize
        
        # 2. Initialize Face Capture
        print("\n[2/3] Starting Face Capture...")
        self.face_capture = PiFaceCapture()
        
        # Set callbacks
        self.face_capture.on_match_result = self._handle_match_result
        self.face_capture.on_images_received = self._handle_images_received
        
        # Load models
        self.face_capture.load_models()
        
        # 3. Connect to laptop
        print("\n[3/3] Connecting to Laptop...")
        connected = self.face_capture.connect_to_laptop()
        
        if not connected:
            print("\n⚠️  WARNING: Not connected to laptop")
            print("   The system will still work, but cannot send/receive data")
            print("   Make sure laptop is running and reachable at:")
            print(f"   IP: {LAPTOP_IP}")
            print(f"   Port: {LAPTOP_PORT}")
            
            response = input("\nContinue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Exiting...")
                sys.exit(0)
        
        print("\n" + "="*60)
        print("✓ PI SYSTEM READY")
        print("="*60)
        print(f"Display: Running in slideshow mode")
        print(f"Camera: Press SPACE to capture, Q to quit")
        print(f"Network: {'Connected' if connected else 'Disconnected'}")
        print("="*60 + "\n")
    
    def run(self):
        """Run main application."""
        self.running = True
        
        try:
            # Start camera (blocking call)
            self.face_capture.start_camera()
            self.face_capture.run()
            
        except KeyboardInterrupt:
            print("\n[SYSTEM] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Shutdown all modules."""
        self.running = False
        
        print("\n[SYSTEM] Shutting down...")
        
        # Stop face capture
        if self.face_capture:
            print("[SYSTEM] Stopping face capture...")
            self.face_capture.stop()
        
        # Stop display
        if self.display_manager:
            print("[SYSTEM] Stopping display...")
            self.display_manager.stop()
        
        print("[SYSTEM] Shutdown complete")
    
    def _handle_match_result(self, msg: dict):
        """
        Handle match result from laptop.
        Called by face_capture module.
        """
        hit = msg.get('hit', False)
        person_id = msg.get('person_id', 'unknown')
        score = msg.get('score', 0.0)
        
        if hit:
            print(f"\n🎯 MATCH: Person #{person_id} (confidence: {score:.1%})")
            self.waiting_for_match = True
        else:
            print(f"\n❌ NO MATCH (best score: {score:.1%})")
            self.waiting_for_match = False
            
            # Return display to slideshow if not already
            if self.display_manager:
                self.display_manager.return_to_slideshow()
    
    def _handle_images_received(self, images: list):
        """
        Handle images received from laptop.
        Called by face_capture module.
        """
        if not self.waiting_for_match:
            print("[WARNING] Received images but not waiting for match")
            return
        
        print(f"\n📥 IMAGES RECEIVED: {len(images)} images")
        
        # Get match info from last result
        if self.face_capture and self.face_capture.network_client:
            # Extract from last message (stored in network client)
            # For now, use placeholder values
            person_id = "MATCHED"
            score = 0.90
            
            # Trigger display
            if self.display_manager:
                self.display_manager.trigger_match(
                    person_id=person_id,
                    score=score,
                    images_data=images
                )
                print("✓ Display triggered")
        
        self.waiting_for_match = False


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point for Pi application."""
    
    # Check Python version
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        sys.exit(1)
    
    # Check required directories
    print("Checking system requirements...")
    
    # Check stock images
    if not STOCK_IMAGES_DIR.exists():
        print(f"\n⚠️  Stock images directory not found: {STOCK_IMAGES_DIR}")
        print("Creating directory...")
        STOCK_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {STOCK_IMAGES_DIR}")
        print("\n📌 IMPORTANT: Please add slideshow images to this directory:")
        print(f"   {STOCK_IMAGES_DIR.absolute()}")
        print("   Supported formats: JPG, PNG, BMP")
        print("")
    
    # Check models
    if not FACE_DETECTOR_MODEL.exists():
        print(f"\n❌ ERROR: Face detector model not found")
        print(f"   Expected: {FACE_DETECTOR_MODEL}")
        print("\n📌 Please install InsightFace models:")
        print("   1. Install insightface: pip install insightface")
        print("   2. Download buffalo_l models:")
        print("      from insightface.app import FaceAnalysis")
        print("      app = FaceAnalysis(name='buffalo_l')")
        print("      app.prepare(ctx_id=0)")
        print("")
        sys.exit(1)
    
    if not FACE_RECOGNIZER_MODEL.exists():
        print(f"\n❌ ERROR: Face recognizer model not found")
        print(f"   Expected: {FACE_RECOGNIZER_MODEL}")
        print("\n📌 Please install InsightFace models (see above)")
        print("")
        sys.exit(1)
    
    print("✓ All requirements satisfied\n")
    
    # Create and run application
    app = PiApplication()
    app.initialize()
    app.run()


if __name__ == "__main__":
    main()
