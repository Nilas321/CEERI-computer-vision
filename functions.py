import cv2
import numpy as np
from collections import deque
import time

tracked_objects = {}  # Dictionary to hold tracked objects
_id_counter = 0  # Global counter for object IDs
STABILITY_THRESHOLD = 5  # Number of frames an object must be stable to be considered as new object 

def create_median_background(cap):
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
    return np.median(background_frames, axis=0).astype(np.uint8)

def send_string_with_delay(string, delay_seconds):
        time.sleep(delay_seconds)  # Delay sending
        return string

def track_objects(contours, frame):
    global _id_counter
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
    for obj_id, [prev_cx, prev_cy, _, stable_count,cnt] in tracked_objects.items():
        # Find nearest in new frame
        distances = [np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                      for (cx, cy, _) in objects]
        if distances:
            min_idx = np.argmin(distances)
            if distances[min_idx] < 20:  # Max allowed displacement
                new_cx, new_cy, new_cnt = objects[min_idx]
                tracked_objects[obj_id][0] = new_cx  # Update X
                tracked_objects[obj_id][1] = new_cy  # Update Y
                tracked_objects[obj_id][3] += 1      # Increment stability count
                tracked_objects[obj_id][4] = new_cnt  # Update contour
                
                del objects[min_idx]
                continue  # Continue to next tracked object
  # Exit loop after updating this object
            else:
                tracked_objects[obj_id] = [prev_cx, prev_cy, False,-1,None]

    for obj_id in list(tracked_objects.keys()):
        if not tracked_objects[obj_id][2]:
            del tracked_objects[obj_id]  # Remove objects that are not found in current frame
    # Add new objects
    for cx, cy, cnt in objects:
        new_id = _id_counter
        _id_counter += 1
        tracked_objects[new_id] = [cx, cy, True, 0, cnt]  # Added contour storage
    
    return tracked_objects

#Function to detect size and color of objects if object is stable
def detect_object_properties(tracked_objects, frame):
    object_properties = {}
    for obj_id, (cx, cy, is_stable, stable_count, cnt) in tracked_objects.items():
        if is_stable and stable_count >= STABILITY_THRESHOLD:
            # Calculate size
            area = cv2.contourArea(cnt)
            #size is small if area is greater than 500 and smaller than 2000, medium if greater than 2000 and smaller than 5000, large if greater than 5000
            size = "Large" if area > 5000 else "Medium" if 2000 < area <= 5000 else "Small"

            # Calculate color
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=mask)[:3]  # BGR format
            #convert to hsv color space
            # Convert mean color from BGR to HSV
            mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
            hue = mean_color_hsv[0]

            # Define color ranges in HSV (OpenCV hue range: 0-179)
            if hue < 10 or hue >= 160:
                color = "Red"
            elif 10 <= hue < 20:
                color = "Orange"
            elif 20 <= hue < 30:
                color = "Yellow"
            elif 30 <= hue < 85:
                color = "Green"
            elif 85 <= hue < 130:
                color = "Blue"
            else:
                color = "Purple"

            # Store properties           
            object_properties[obj_id] = {
                'size': size,
                'color': color,
                'position': (cx, cy)
            }
    
    return object_properties

#given coordinate of object (x,y,w,h) and coordinates in frame ,x1,x2,x3 create a function that returns in which splice of x axis it is in as lane 1 lane2 lane3 

def get_lane(x,y,w,h):
    x1 = 151
    x2 = 291
    x3 = 427

    lane = []

    x_end = x + w

    # Only detect lane if object intersects y = 35
    if y <= 35 <= y + h:
        if x1 > x >= 27:
            if x1 > x_end:
                lane.append(1)
            elif x2 > x_end:
                lane.extend([1, 2])
            elif x3 > x_end:
                lane.extend([1, 2, 3])
            else:
                lane.extend([1, 2, 3])
        elif x2 > x:
            if x2 > x_end:
                lane.append(2)
            elif x3 > x_end:
                lane.extend([2, 3])
            else:
                lane.extend([2, 3])
        elif x3 > x:
            if x3 > x_end:
                lane.append(3)
            else:
                lane.append(3)
        else:

            # Outside all defined lane regions
            pass
    return lane

#a function that outputs lane info, size and color info on the frame 

def annotate_frame(frame, tracked_objects, object_properties):
    for obj_id, (cx, cy, is_stable, stable_count, cnt) in tracked_objects.items():
        if is_stable and stable_count >= STABILITY_THRESHOLD:
            size = object_properties[obj_id]['size']
            color = object_properties[obj_id]['color']
            position = object_properties[obj_id]['position']
            x, y, w, h = cv2.boundingRect(cnt)
            lane = get_lane(x,y,w,h)  # Get lane info from contour
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Annotate size and color
            cv2.putText(frame, f"ID: {obj_id}, Size: {size}, Color: {color}, Lane: {lane}", 
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

#run the main function to capture video and process frames using the above functions

def main():
    global tracked_objects
    cap = cv2.VideoCapture(0)  # Change to your video source
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    background = create_median_background(cap)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Subtract background
        diff = cv2.absdiff(blurred, cv2.cvtColor(background, cv2.COLOR_BGR2GRAY))
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Track objects
        tracked_objects = track_objects(contours, frame)

        # Detect object properties
        object_properties = detect_object_properties(tracked_objects, frame)

        # Annotate frame with object info
        annotated_frame = annotate_frame(frame, tracked_objects, object_properties)

        # Display the resulting frame
        cv2.imshow('Object Tracking', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()