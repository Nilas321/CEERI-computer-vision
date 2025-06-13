import cv2

# Open the default webcam (index 0)
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

ret, frame = cap.read()
if ret:
    # Save the image in the same directory as the script
    cv2.imwrite('image.jpg', frame)
    print("Image saved as captured_image.jpg in the script's directory.")
else:
    print("Failed to capture image.")

cap.release()
cv2.destroyAllWindows()
