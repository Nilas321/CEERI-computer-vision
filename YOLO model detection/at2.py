import cv2
import numpy as np
from roboflow import Roboflow

# --- Initialize Roboflow Model ---
rf = Roboflow(api_key="rbLYmnZ8sDQ2bre3Y4r7")
project = rf.workspace().project("plastic-recyclable-detection")
model = project.version(1).model

GAUSSIAN_BLUR = (5, 5)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame temporarily for Roboflow (alternatively, convert to PIL and use API buffer)
    temp_path = "temp_frame.jpg"
    cv2.imwrite(temp_path, frame)

    # Run inference
    result = model.predict(temp_path, confidence=40, overlap=30).json()

    valid_contours = []

    for prediction in result['predictions']:
        x, y, w, h = int(prediction['x']), int(prediction['y']), int(prediction['width']), int(prediction['height'])
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2

        roi = frame[y1:y2, x1:x2]

        # Preprocess ROI for contours
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR, 0)
        _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:
                cnt[:, :, 0] += x1
                cnt[:, :, 1] += y1
                valid_contours.append(cnt)

    for cnt in valid_contours:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_color = cv2.mean(frame, mask=mask)[:3]

        # Convert to HSV
        mean_color_hsv = cv2.cvtColor(np.uint8([[mean_color]]), cv2.COLOR_BGR2HSV)[0][0]
        hue = mean_color_hsv[0]

        # Color classification
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

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{color}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        area = cv2.contourArea(cnt)
        if area > 5000:
            size = "Large"
        elif 2000 < area <= 5000:
            size = "Medium"
        else:
            size = "Small"
        cv2.putText(frame, f"{size}", (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

    cv2.imshow('Roboflow YOLOv8 Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
