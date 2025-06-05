import cv2

cap = cv2.VideoCapture(0)  # Use camera or video stream

# Use mp4 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('background.mp4', fourcc, 20.0, (640, 480))

print("Recording background... Press 'q' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    cv2.imshow('Recording background', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Background video saved as 'background.mp4'")
