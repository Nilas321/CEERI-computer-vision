import cv2
import numpy as np

# Open the default webcam (camera index 0)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame from the video stream
    _, frame = cap.read()
    
    # Select the region of interest (ROI) from the frame
    # Here, we extract a rectangle defined by (x1=210, y1=194) to (x2=590, y2=438)
    req_area = frame[194:438, 210:590]
    
    # Convert the ROI to grayscale for easier processing
    gray_frame = cv2.cvtColor(req_area, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise and smooth the image before thresholding
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    
    # Binarize the blurred grayscale image using a fixed threshold
    # Pixels >200 become 255 (white), others become 0 (black)
    _, threshold = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)
    #threshold=cv2.bitwise_not(threshold)
    # Find contours (continuous white blobs) in the binarized image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        # Create a mask for the current contour
        mask = np.zeros(req_area.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # Fill the contour area with white
        
        # Calculate the mean color within the contour area of the ROI
        mean_color = cv2.mean(req_area, mask=mask)[:3]
        print(f"Mean color in rectangle: {mean_color}")
        
        # Draw a filled rectangle (50x50) at the top-left corner of the frame,
        # colored with the mean color found in the contour area
        rect_top_left = (0, 0)
        rect_bottom_right = (50, 50)
        mean_color_int = tuple(map(int, mean_color))
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, mean_color_int, thickness=-1)
        
        # Draw a green bounding rectangle around the detected contour in the ROI
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(req_area, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # Display the original frame, the ROI, and the binarized image
    cv2.imshow("Frame", frame)
    cv2.imshow("Required Area", req_area)
    cv2.imshow("Binarized Frame", threshold)
    
    # Wait for 1 ms for a key press; break loop if 'Enter' (ASCII 13) is pressed
    key = cv2.waitKey(1)
    if key == 13:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
# Note: The code captures video from the webcam, processes a specific region of interest,
# and displays the results in real-time. The mean color of detected contours is used to fill a rectangle