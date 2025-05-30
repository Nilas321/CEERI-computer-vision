import cv2
import numpy as np

# Read video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 for default camera

# Collect first 120 frames to estimate the background
background_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)

# Compute median background
median_background = np.median(background_frames, axis=0).astype(np.uint8)
cv2.imshow('Median Background', median_background)

# Object tracking setup
tracked_objects = {}  # id: {'contour': cnt, 'position': (x, y), 'missed_frames': 0}
next_id = 0
max_missed_frames = 10

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Foreground mask
    diff = cv2.absdiff(frame, median_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, foreground_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

    # Match current contours with tracked objects
    matched_ids = set()
    for curr_cnt in valid_contours:
        x2, y2, w2, h2 = cv2.boundingRect(curr_cnt)
        matched = False
        for obj_id, data in tracked_objects.items():
            x1, y1, w1, h1 = cv2.boundingRect(data['contour'])
            if (abs(x1 - x2) < 20 and abs(y1 - y2) < 20 and
                abs(w1 - w2) < 20 and abs(h1 - h2) < 20):
                tracked_objects[obj_id]['contour'] = curr_cnt
                tracked_objects[obj_id]['position'] = (x2, y2)
                tracked_objects[obj_id]['missed_frames'] = 0
                matched_ids.add(obj_id)
                matched = True
                break
        if not matched:
            tracked_objects[next_id] = {
                'contour': curr_cnt,
                'position': (x2, y2),
                'missed_frames': 0
            }
            matched_ids.add(next_id)
            next_id += 1

    # Increment missed frames for unmatched objects
    to_delete = []
    for obj_id in tracked_objects:
        if obj_id not in matched_ids:
            tracked_objects[obj_id]['missed_frames'] += 1
            if tracked_objects[obj_id]['missed_frames'] > max_missed_frames:
                to_delete.append(obj_id)

    # Remove lost objects
    for obj_id in to_delete:
        del tracked_objects[obj_id]

    # Display tracked objects with mean color boxes
    count = 0
    for obj_id, data in tracked_objects.items():
        cnt = data['contour']
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)[:3]
        mean_color_int = tuple(map(int, mean_color))

        rect_top_left = (0, 0 + 50 * count)
        rect_bottom_right = (50, 50 + 50 * count)
        cv2.putText(frame, f"box {count} (ID:{obj_id})", (55, 50 + 50 * count), 1, 1, (255, 255, 0))
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, mean_color_int, thickness=-1)
        count += 1

        # Draw bounding box with size category
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 500 < area < 1000:
            color = (255, 0, 0)
            label = "Small"
        elif 1000 < area < 2000:
            color = (0, 255, 0)
            label = "Medium"
        else:
            color = (0, 0, 255)
            label = "Large"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}, ID:{obj_id}", (x, y), 1, 1, (255, 255, 0))

    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)

    if cv2.waitKey(1) == 13:  # Enter key
        break

cap.release()
cv2.destroyAllWindows()
