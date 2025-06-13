import cv2
import numpy as np

tracked_objects = {}  # key: ID, value: {'pos': (x, y, w, h), 'stable_count': int, 'mean_color': (B, G, R)}
next_id = 0
STABILITY_THRESHOLD = 12
AREA_THRESHOLD = 500

def is_same_position(pos1, pos2, tolerance=10):
    x1, y1, w1, h1 = pos1
    x2, y2, w2, h2 = pos2
    return abs(x1 - x2) < tolerance and abs(y1 - y2) < tolerance

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Collect first 120 frames to calculate background
background_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)

median_background = np.median(background_frames, axis=0).astype(np.uint8)
cv2.imshow('Median Background', median_background)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    diff = cv2.absdiff(frame, median_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, foreground_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    updated_objects = {}
    display_box_index = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < AREA_THRESHOLD:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        current_pos = (x, y, w, h)
        matched_id = None

        for obj_id, data in tracked_objects.items():
            if is_same_position(current_pos, data['pos']):
                matched_id = obj_id
                break

        # Create mask for mean color calc
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)

        if matched_id is not None:
            old_data = tracked_objects[matched_id]
            stable = is_same_position(current_pos, old_data['pos'])
            stable_count = old_data['stable_count'] + 1 if stable else 0

            # Update only after stability threshold
            if stable_count >= STABILITY_THRESHOLD:
                mean_color = old_data['mean_color']
            else:
                mean_color = cv2.mean(frame, mask=mask)[:3]

            updated_objects[matched_id] = {
                'pos': current_pos,
                'stable_count': stable_count,
                'mean_color': mean_color
            }
        else:
            # New object
            mean_color = cv2.mean(frame, mask=mask)[:3]
            updated_objects[next_id] = {
                'pos': current_pos,
                'stable_count': 0,
                'mean_color': mean_color
            }
            matched_id = next_id
            next_id += 1

    # Draw objects
    for obj_id, data in updated_objects.items():
        (x, y, w, h) = data['pos']
        mean_color = tuple(map(int, data['mean_color']))
        stable_count = data['stable_count']

        # Area-based size label (only after stability)
        if stable_count >= STABILITY_THRESHOLD:
            area = w * h
            if area < 1000:
                color = (255, 0, 0)
                label = "Small"
            elif area < 2000:
                color = (0, 255, 0)
                label = "Medium"
            else:
                color = (0, 0, 255)
                label = "Large"
        else:
            color = (0, 255, 255)  # Yellow for unstable
            label = "Detecting..."

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}, box {obj_id}", (x, y), 1, 1, (255, 255, 0))

        #Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the current contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # Fill the contour area with white
        
        # Calculate the mean color within the contour area of the ROI
        mean_color = cv2.mean(hsv_frame, mask=mask)[:3]
        #print(f"Mean color in rectangle {count}: {mean_color}")
        h=mean_color[0]
        s=mean_color[1]
        v=mean_color[2]
        if (10>h>=0 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Red(Lower)")
        elif (179>h>=160 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Red(Upper)")
        elif (25>h>=10 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Orange")
        elif (35>=h>=25 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Yellow")
        elif (85>=h>=36 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Green")
        elif (100>=h>=86 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Cyan")
        elif (130>=h>=101 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Blue")
        elif (160>=h>=131 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {obj_id}: Purple")
        elif (170>=h>=145 and 255>=s>=50 and 255>=v>=150):
            print(f"Mean Colour in the rectange {obj_id}: Pink")
        elif (20>=h>=10 and 255>=s>=100 and 200>=v>=20):
            print(f"Mean Colour in the rectange {obj_id}: Brown")
        elif (180>=h>=0 and 30>=s>=0 and 255>=v>=200):
            print(f"Mean Colour in the rectange {obj_id}: White")
        elif (180>=h>=0 and 50>=s>=0 and 200>=v>=50):
            print(f"Mean Colour in the rectange {obj_id}: Grey")
        elif (180>=h>=0 and 255>=s>=0 and 50>=v>=0):
            print(f"Mean Colour in the rectange {obj_id}: Black")

       # Display mean color box
        '''top_left = (0, 0 + 50 * display_box_index)
        bottom_right = (50, 50 + 50 * display_box_index)
        cv2.rectangle(frame, top_left, bottom_right, mean_color, -1)
        cv2.putText(frame, f"box {obj_id}", (55, 50 + 50 * display_box_index), 1, 1, (255, 255, 0))
        display_box_index += 1'''
    tracked_objects = updated_objects
    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)

    key = cv2.waitKey(1)
    if key == 13:  # Enter key to exit
        break

cap.release()
cv2.destroyAllWindows()