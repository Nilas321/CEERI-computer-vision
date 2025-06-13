import cv2
import numpy as np

#capture video
cap=cv2.VideoCapture(0) #Opens the webcam. The argument 0 selects the default camera.

while True:
    _,frame=cap.read() #capture frames from a video source_Returns a tuple containing two values
                       #boolean(indicating whether the frame captured successfully)_frame(NumPy array representing the captured image)'''
                       #_ contains the boolean value, frame contains the image array


# Convert to grayscale
gray_bg = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
gray_obj = cv2.cvtColor(object_frame, cv2.COLOR_BGR2GRAY)

# Subtract the background from the object image
diff = cv2.absdiff(gray_obj, gray_bg)

# Threshold to get the foreground mask
_, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# Optional: Apply morphology to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Extract the object
foreground = cv2.bitwise_and(object_frame, object_frame, mask=mask)

cv2.imshow("Foreground", foreground)
cv2.waitKey(0)
cv2.destroyAllWindows()
