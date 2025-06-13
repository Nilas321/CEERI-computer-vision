import cv2
import numpy as np
from collections import deque

# HSV ranges for common plastics (customize based on calibration)
COLOR_RANGES = {
    "PET_clear": ([0, 0, 200], [180, 30, 255]),
    "HDPE_white": ([0, 0, 150], [180, 50, 255]),
    "PP_black": ([0, 0, 0], [180, 255, 50])
}

# Temporal smoothing buffers
color_history = {}
size_history = {}

def process_frame(frame):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Dynamic brightness adaptation
    v_mean = np.mean(hsv[:,:,2])
    if abs(v_mean - process_frame.prev_v) > 15:
        adjust_hsv_ranges(v_mean)
    process_frame.prev_v = v_mean
    
    # Multi-color masking
    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for color_name, (lower, upper) in COLOR_RANGES.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Contour detection
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

process_frame.prev_v = 0

def adjust_hsv_ranges(v_mean):
    global COLOR_RANGES
    # Adjust brightness thresholds based on mean V value
    for color_name, (lower, upper) in COLOR_RANGES.items():
        lower[2] = max(0, min(255, lower[2] + (v_mean - 128) // 2))
        upper[2] = max(0, min(255, upper[2] + (v_mean - 128) // 2))
        COLOR_RANGES[color_name] = (lower, upper)
def track_objects(contours, frame):
    global color_history, size_history
    
    # Convert contours to tracking format
    objects = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"])
        cy = int(M["m01"]/M["m00"])
        objects.append((cx, cy, cnt))
    
    # Simple centroid tracking
    tracked = {}
    for obj_id, (prev_cx, prev_cy) in tracked.items():
        # Predict position using treadmill speed
        predicted_x = prev_cx 
        # Find nearest in new frame
        distances = [np.sqrt((cx-predicted_x)**2 + (cy-prev_cy)**2) 
                    for cx, cy, _ in objects]
        if distances:
            min_idx = np.argmin(distances)
            if distances[min_idx] < 50:  # Max allowed displacement
                tracked[obj_id] = objects[min_idx][:2]
                del objects[min_idx]
    
    # Handle new objects
    for cx, cy, cnt in objects:
        new_id = max(tracked.keys(), default=0) + 1
        tracked[new_id] = (cx, cy)
        
        # Initialize history buffers
        color_history[new_id] = deque(maxlen=30)
        size_history[new_id] = deque(maxlen=30)
    
    return tracked

def get_stabilized_measurements(cnt, obj_id):
    # Current measurements
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Size smoothing
    size_history[obj_id].append((area, perimeter))
    smoothed_area = np.mean([a for a, p in size_history[obj_id]])
    smoothed_perim = np.mean([p for a, p in size_history[obj_id]])
    
    # Color averaging
    mask = np.zeros_like(frame)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask)[:3]
    color_history[obj_id].append(mean_color)
    avg_color = np.mean(color_history[obj_id], axis=0)
    
    return smoothed_area, smoothed_perim, avg_color

cap = cv2.VideoCapture(0)  # Webcam input
tracked_objects = {}
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    contours = process_frame(frame)
    tracked = track_objects(contours, frame)
    
    # Visualization and logging
    for obj_id, (cx, cy) in tracked.items():
        cnt = [c for c in contours if (cx, cy) in c][0]
        area, perim, color = get_stabilized_measurements(cnt, obj_id)
        
        # Draw stabilized measurements
        cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
        cv2.putText(frame, f"ID:{obj_id}", (cx-20, cy-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    cv2.imshow("Plastic Detection", frame)
    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
# Note: This code is a simplified version and may require adjustments based on your specific setup and requirements.
# Ensure you have OpenCV installed: pip install opencv-python