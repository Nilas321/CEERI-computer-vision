import cv2
import numpy as np

# --- Define a class to track objects ---
class TrackedObject:
    def __init__(self, bbox, obj_id):
        self.bbox = bbox  # (x, y, w, h)
        self.frames_stable = 1
        self.color = None
        self.id = obj_id

    def update(self, new_bbox):
        self.bbox = new_bbox
        self.frames_stable += 1

# --- Helper function to check similarity between bounding boxes ---
def is_similar(bbox1, bbox2, thresh=20):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return (
        abs(x1 - x2) < thresh and abs(y1 - y2) < thresh and
        abs(w1 - w2) < thresh and abs(h1 - h2) < thresh
    )

# --- Initialize ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
background_frames = []

# Collect first 120 frames for median background
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)

median_background = np.median(background_frames, axis=0).astype(np.uint8)
cv2.imshow('Median Background', median_background)

# Tracking variables
tracked_objects = []
next_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    diff = cv2.absdiff(frame, median_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, foreground_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
    current_bboxes = [cv2.boundingRect(cnt) for cnt in valid_contours]

    # --- Match current bounding boxes to tracked objects ---
    updated_ids = set()
    for curr_bbox in current_bboxes:
        matched = False
        for obj in tracked_objects:
            if is_similar(obj.bbox, curr_bbox):
                obj.update(curr_bbox)
                updated_ids.add(obj.id)
                matched = True
                break
        if not matched:
            tracked_objects.append(TrackedObject(curr_bbox, next_id))
            next_id += 1

    # --- Remove stale objects ---
    tracked_objects = [obj for obj in tracked_objects if obj.frames_stable >= 10 or obj.id in updated_ids]

    # --- Draw stable objects and display mean color ---
    count = 0
    for obj in tracked_objects:
        if obj.frames_stable >= 10:
            x, y, w, h = obj.bbox

            if obj.color is None:
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                mean_color = cv2.mean(frame, mask=mask)[:3]
                obj.color = tuple(map(int, mean_color))

            cv2.rectangle(frame, (x, y), (x + w, y + h), obj.color, 2)
            cv2.putText(frame, f"Obj {obj.id} (Stable)", (x, y - 10), 1, 1, (255, 255, 0), 2)

            # Draw color patch on side
            rect_top_left = (0, 0 + 50 * count)
            rect_bottom_right = (50, 50 + 50 * count)
            cv2.rectangle(frame, rect_top_left, rect_bottom_right, obj.color, thickness=-1)
            cv2.putText(frame, f"Obj {obj.id}", (55, 40 + 50 * count), 1, 1, (255, 255, 0))
            count += 1

    # --- Show frames ---
    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)

    if cv2.waitKey(1) == 13:  # Enter key to break
        break

cap.release()
cv2.destroyAllWindows()