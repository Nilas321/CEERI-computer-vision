import cv2
from ultralytics import YOLO
import torch
import numpy as np
# YOLOv5 Real-Time Object Detection Example

# 1. Initialize YOLOv5 model (automatically downloads if missing)
model = YOLO('yolov5nu.pt')  # Options: yolov5n.pt (fastest), yolov5x.pt (most accurate)
GAUSSIAN_BLUR = (5, 5)
# 2. Configure video input source
cap = cv2.VideoCapture(0)  # Webcam (0), video file path, or RTSP URL

# 3. Performance optimizations
model.fuse()  # Fuse Conv2d + BatchNorm layers
if torch.cuda.is_available():
    model.to('cuda')  # GPU acceleration

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. Run YOLOv5 inference
    results = model(frame, verbose=False)  # Set verbose=False to disable logs
    
    # 5. Process and visualize results
    annotated_frame = results[0].plot()  # Auto-draw boxes/labels
    
    for box in results[0].boxes:
        if box.conf < 0.5:  # Confidence threshold
            continue
            
        # 1. Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        roi = frame[y1:y2, x1:x2]
        
        # 2. Preprocess for contour detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
        
        # 3. Find contours in ROI
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        for cnt in valid_contours:
            cnt[:, :, 0] += x1  # Shift x-coordinates
            cnt[:, :, 1] += y1  # Shift y-coordinates
    #find colour of object
    for cnt in valid_contours:
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

        print(f"Detected color: {color}")
    # 6. Display output
    cv2.imshow('YOLOv5 Real-Time Detection', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()