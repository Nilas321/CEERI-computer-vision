import cv2
import numpy as np
from collections import deque

# HSV ranges for common plastics (customize based on calibration)
COLOR_RANGES = color_ranges_hsv = {
    "red_1":       ([0, 100, 100], [10, 255, 255]),     # Lower red range
    "red_2":       ([160, 100, 100], [179, 255, 255]),  # Upper red range (due to hue wrap)

    "orange":      ([10, 100, 100], [25, 255, 255]),

    "yellow":      ([25, 100, 100], [35, 255, 255]),

    "green":       ([36, 100, 100], [85, 255, 255]),

    "cyan":        ([86, 100, 100], [100, 255, 255]),

    "blue":        ([101, 100, 100], [130, 255, 255]),

    "purple":      ([131, 100, 100], [160, 255, 255]),

    "pink":        ([145, 50, 150], [170, 255, 255]),

    "brown":       ([10, 100, 20], [20, 255, 200]),

    "white":       ([0, 0, 200], [180, 30, 255]),        # Very low saturation, high value

    "gray":        ([0, 0, 50], [180, 50, 200]),         # Low saturation, medium brightness

    "black":       ([0, 0, 0], [180, 255, 50])           # Low value
}
#temporary color ranges for testing 
# color_ranges_hsv = {
#     # Clear & White Plastics
#     "PET_clear":        ([0, 0, 200], [180, 30, 255]),      # Transparent
#     "HDPE_white":       ([0, 0, 150], [180, 50, 255]),      # Opaque white

#     # Black Plastics
#     "PP_black":         ([0, 0, 0], [180, 255, 50]),        # Black

#     # Red Plastics
#     "PP_red_1":         ([0, 100, 100], [10, 255, 255]),    # Red range part 1
#     "PP_red_2":         ([160, 100, 100], [179, 255, 255]), # Red range part 2

#     # Orange Plastics
#     "HDPE_orange":      ([10, 100, 100], [20, 255, 255]),

#     # Yellow Plastics
#     "HDPE_yellow":      ([20, 100, 100], [35, 255, 255]),

#     # Green Plastics
#     "HDPE_green":       ([35, 100, 100], [85, 255, 255]),

#     # Blue Plastics
#     "HDPE_blue":        ([100, 100, 100], [130, 255, 255]),

#     # Purple Plastics
#     "HDPE_purple":      ([130, 100, 100], [160, 255, 255]),

#     # Gray Plastics (low saturation)
#     "PP_gray":          ([0, 0, 50], [180, 50, 200]),

#     # Brown Plastics
#     "PP_brown":         ([10, 100, 20], [20, 255, 200]),

#     # Skin-tone Plastics (e.g., cosmetic bottles)
#     "ABS_skin":         ([5, 50, 100], [20, 150, 255])
# }

# Temporal smoothing buffers
color_history = {}
size_history = {}

# def process_frame(frame):
#     # Convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
#     # Dynamic brightness adaptation
#     v_mean = np.mean(hsv[:,:,2])
#     if abs(v_mean - process_frame.prev_v) > 15:
#         adjust_hsv_ranges(v_mean)
#     process_frame.prev_v = v_mean
    
#     # Multi-color masking
#     combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#     for color_name, (lower, upper) in COLOR_RANGES.items():
#         mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
#         combined_mask = cv2.bitwise_or(combined_mask, mask)
    
#     # Contour detection
#     contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

#     process_frame.prev_v = 0  # Initialize previous brightness value

# def adjust_hsv_ranges(v_mean):
#     global COLOR_RANGES
#     # Adjust the brightness threshold based on the mean value
#     brightness_threshold = max(50, min(255, int(v_mean * 0.5)))
    
#     for color_name, (lower, upper) in COLOR_RANGES.items():
#         lower[2] = max(lower[2], brightness_threshold)
#         upper[2] = min(upper[2], 255)
#         COLOR_RANGES[color_name] = (lower, upper)

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
        predicted_x = prev_cx  # + int(treadmill_speed * frame_time)
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

def get_stabilized_measurements(cnt, obj_id,frame):
    # Current measurements
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    
    # Size smoothing
    size_history[obj_id].append((area, perimeter))
    smoothed_area = np.mean([a for a, p in size_history[obj_id]])
    smoothed_perim = np.mean([p for a, p in size_history[obj_id]])
    
    # Color averaging
    mask = np.zeros_like(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    mean_color = cv2.mean(frame, mask=mask)[:3]
    color_history[obj_id].append(mean_color)
    avg_color = np.mean(color_history[obj_id], axis=0)
    
    return smoothed_area, smoothed_perim, avg_color

cap = cv2.VideoCapture(0)  # Webcam input
tracked_objects = {}
frame_count = 0

background_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)

# Compute median background
median_background = np.median(background_frames, axis=0).astype(np.uint8)

#display the median background
cv2.imshow('Median Background', median_background)

while True:
    ret, frame = cap.read()
    if not ret: break
    
   # Compute absolute difference between current frame and static background
    diff = cv2.absdiff(frame, median_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #apply gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the blurred image to create a binary mask
    _, foreground_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            valid_contours.append(cnt)

    tracked = track_objects(valid_contours, frame)

    # Visualization and logging
    for obj_id, (cx, cy) in tracked.items():
        cnt = [c for c in valid_contours if (cx, cy) in c][0]
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
    
        # Size smoothing
        size_history[obj_id].append((area, perimeter))
        smoothed_area = np.mean([a for a, p in size_history[obj_id]])
        smoothed_perim = np.mean([p for a, p in size_history[obj_id]])
    
    # Color averaging
        mask = np.zeros_like(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)[:3]
        color_history[obj_id].append(mean_color)
        avg_color = np.mean(color_history[obj_id], axis=0)
        
        # Draw stabilized measurements
        cv2.drawContours(frame, [cnt], -1, (0,255,0), 2)
        cv2.putText(frame, f"ID:{obj_id}", (cx-20, cy-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    
    
    cv2.imshow("Plastic Detection", frame)
    if cv2.waitKey(1) == 13:  # Enter key
        break


cap.release()
cv2.destroyAllWindows()