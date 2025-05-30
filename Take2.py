import cv2
import numpy as np
from collections import deque

count = 0
# Temporal smoothing buffers
color_history = {}
size_history = {}
tracked_objects = {}
# Read video
cap = cv2.VideoCapture(0)  # 0 for default camera
frame_count = 0

# Collect first 120 frames
background_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)
    frame_count += 1

# Compute median background
median_background = np.median(background_frames, axis=0).astype(np.uint8)

#display the median background
cv2.imshow('Median Background', median_background)

#function to track objects
def track_objects(contours, frame):
    global color_history, size_history,tracked_objects
    objects = []
    #convert contours to list that can be tracked
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            objects.append((cx, cy, cnt))

    # If no objects detected, return empty tracked dict
    if not objects:
        return tracked_objects

    # Check for object tracking
    for obj_id, (prev_cx, prev_cy) in tracked_objects.items():
        # Find nearest in new frame
        distances = [np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                      for (cx, cy, _) in objects]
        if distances:
            min_idx = np.argmin(distances)
            if distances[min_idx] < 50:  # Max allowed displacement
                tracked_objects[obj_id] = objects[min_idx][:2]
                del objects[min_idx]

    for cx, cy, cnt in objects:
        new_id = max(tracked_objects.keys(), default=0) + 1
        tracked_objects[new_id] = (cx, cy)

        # Initialize color buffers
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)[:3]
        color_history[new_id] = mean_color
        avg_color = np.mean(color_history[new_id], axis=0)
        
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        size_history[new_id] = (area, perimeter)

    return tracked_objects

while frame_count > 119:
    ret, frame = cap.read()
    if not ret:
        break

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
    # Track objects
    tracked_objects = track_objects(valid_contours, frame)
    for cnt in valid_contours:
    #for cnt in valid_contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 500<area<1000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame,"Small,box number" + str(count),(x,y),1,1,(255,255,0))
        elif 1000<area<2000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,"Medium,box number" + str(count),(x,y),1,1,(255,255,0))
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame,"Large,box number" + str(count),(x,y),1,1,(255,255,0))
    # Draw a green bounding rectangle around the detected contour in the ROI
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    # Display the original frame with detected foreground
    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    count = 0
    key = cv2.waitKey(1)
    if key == 13:
        break
    print("Tracked Objects:", tracked_objects)
    print("Color History:", color_history)
    print("Size History:", size_history)
    print("Frame Count:", frame_count)
    print("\n\n")
    frame_count += 1
cap.release()
cv2.destroyAllWindows()
