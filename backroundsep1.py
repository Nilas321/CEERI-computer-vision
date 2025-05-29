import cv2
import numpy as np

# Read video
cap = cv2.VideoCapture(0)  # 0 for default camera
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  )

# Collect first 120 frames
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

while cap.isOpened():
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

    for cnt in contours:
        # Create a mask for the current contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # Fill the contour area with white
        
        # Calculate the mean color within the contour area of the ROI
        mean_color = cv2.mean(frame, mask=mask)[:3]
        print(f"Mean color in rectangle: {mean_color}")
        
        # Draw a filled rectangle (50x50) at the top-left corner of the frame,
        # colored with the mean color found in the contour area
        rect_top_left = (0, 0)
        rect_bottom_right = (50, 50)
        mean_color_int = tuple(map(int, mean_color))
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, mean_color_int, thickness=-1)
        
        # Draw a green bounding rectangle around the detected contour in the ROI
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    # Display the original frame with detected foreground
    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    key = cv2.waitKey(1)
    if key == 13:
        break
cap.release()
cv2.destroyAllWindows()
