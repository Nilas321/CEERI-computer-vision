import cv2
import numpy as np


bg_cap = cv2.VideoCapture("fbg.mp4")
bg_frame_count = int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Read the first background frame (the reference frame for sync)
bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
ret_bg, bg_frame = bg_cap.read()
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Use camera or video stream
bg_synced = False


# Use mp4 codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('background.mp4', fourcc, 20.0, (640, 480))

print("Recording background... Press 'ENTER KEY' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if not bg_synced:
        # Step 1: Check similarity between current real frame and reference background frame
        diff = cv2.absdiff(frame, bg_frame)
        score = np.sum(diff)
        print(score/100000)

        if score < 1200000:  #  Tune this threshold based on your setup
            print("Background synced!")
            bg_synced = True
        else:
            cv2.putText(frame, "Syncing background...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Real Frame", frame)
            cv2.imshow("background",bg_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # Keep checking next live frame

    out.write(frame)
    cv2.imshow('Recording background', frame)

    diff = cv2.absdiff(frame, bg_frame)
    score = np.sum(diff)
    print(score/100000)

    if score < 1100000:  # ✅ Tune this threshold based on your setup
        print('video recorded')
        break

    if cv2.waitKey(1)==13:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Background video saved as 'background.mp4'")
