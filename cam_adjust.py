import cv2
import time

print("Simple Camera Adjustment - Press 'q' to quit")

# Use your working pipeline
pipeline = "libcamerasrc ! video/x-raw,width=1640,height=1232,framerate=30/1 ! videoconvert ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed! Trying simple pipeline...")
    pipeline = "libcamerasrc ! video/x-raw ! videoconvert ! appsink"
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR: Cannot open camera!")
    exit(1)

print("Camera opened! Adjust position, then press 'q' when done")

zoom = 1.0
pan_x = 0
pan_y = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w = frame.shape[:2]
    
    # Apply zoom if needed
    if zoom > 1.0:
        zoom_w = int(w / zoom)
        zoom_h = int(h / zoom)
        center_x = w // 2 + pan_x
        center_y = h // 2 + pan_y
        
        x1 = max(0, center_x - zoom_w // 2)
        y1 = max(0, center_y - zoom_h // 2)
        x2 = min(w, center_x + zoom_w // 2)
        y2 = min(h, center_y + zoom_h // 2)
        
        if x2 > x1 and y2 > y1:
            frame = frame[y1:y2, x1:x2]
            frame = cv2.resize(frame, (w, h))
    
    # Draw center and face area
    cv2.circle(frame, (w//2, h//2), 5, (0, 255, 0), -1)
    cv2.rectangle(frame, (w//2-150, h//2-150), (w//2+150, h//2+150), (0, 255, 0), 2)
    
    # Show controls
    cv2.putText(frame, "Z/X: Zoom, Arrows: Pan, Q: Quit", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Zoom: {zoom:.1f}x  Pan: ({pan_x}, {pan_y})", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Adjust Camera Position', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('z'):
        zoom = min(3.0, zoom + 0.1)
    elif key == ord('x'):
        zoom = max(1.0, zoom - 0.1)
    elif key == 81:  # Left
        pan_x -= 10
    elif key == 83:  # Right
        pan_x += 10
    elif key == 82:  # Up
        pan_y -= 10
    elif key == 84:  # Down
        pan_y += 10

cap.release()
cv2.destroyAllWindows()

print(f"\nFinal settings:")
print(f"Zoom: {zoom}")
print(f"Pan X: {pan_x}")
print(f"Pan Y: {pan_y}")
print("\nUse these values in your face detection code.")
