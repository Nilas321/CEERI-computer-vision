import cv2
import numpy as np

# Initialize background subtractor with optimized parameters
backSub = cv2.createBackgroundSubtractorMOG2(
    history=200,         # Reduced from default 500 for faster adaptation
    varThreshold=32,     # Increased from 16 for better noise immunity
    detectShadows=True
)

cap = cv2.VideoCapture(0)

# Morphological kernel for noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Smoothing parameters
DECAY_FACTOR = 0.9       # For temporal smoothing
smoothed_mask = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply background subtraction with learning rate
    fg_mask = backSub.apply(frame, learningRate=0.001)  # Slow background update

    # Temporal smoothing using weighted average
    if smoothed_mask is None:
        smoothed_mask = fg_mask.astype(float)
    else:
        smoothed_mask = DECAY_FACTOR * smoothed_mask + (1-DECAY_FACTOR) * fg_mask
    temp_smoothed = smoothed_mask.astype(np.uint8)

    # Process mask
    _, mask = cv2.threshold(temp_smoothed, 127, 255, cv2.THRESH_BINARY)  # Keep shadows as 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2) # Fill holes

    # Find and filter contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Advanced filtering
        if (area > 500 and                     # Minimum area
            w > 30 and h > 30 and              # Minimum dimensions
            area/(w*h) > 0.4 and               # Compactness check
            cv2.arcLength(cnt, True) > 100):   # Perimeter check
            valid_contours.append(cnt)

            # Get mean color within contour
            contour_mask = np.zeros_like(mask)
            cv2.drawContours(contour_mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(frame, mask=contour_mask)[:3]
            
            # Draw bounding box at actual contour location
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y-30), (x+50, y), 
                         tuple(map(int, mean_color)), -1)

    # Display results
    cv2.imshow("Frame", frame)
    cv2.imshow("Processed Mask", mask)

    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()
