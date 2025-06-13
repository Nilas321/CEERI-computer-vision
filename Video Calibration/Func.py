import cv2
import numpy as np
from collections import deque
import time

j=0
_id_counter = 1
# Temporal smoothing buffers
color_history = {}
size_history = {}
tracked_objects = {}
detected_object ={}
STABILITY_THRESHOLD = 12
output_string=""
count = 0

#create a function to check if the given frame is same as reference frame, and return the inverse of the boolean value if similar
def is_frame_similar(frame, reference_frame,bg_synced, threshold=2800000):
    diff = cv2.absdiff(frame, reference_frame)
    score = np.sum(diff)
    print(score/100000)

    if score < threshold:
        return not bg_synced
    else:
        return bg_synced


# Load reference frame for background synchronization
bg_frame = cv2.imread("image.jpg")
bg_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)

# Open real-time camera
real_cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

index = 0
bg_synced = False
Waitframes = 120  # seconds to wait for the background to sync
bg_list = []

while True:
    ret_real, real_frame = real_cap.read()
    if not ret_real:
        break

    cv2.imshow("Real-time Camera Frame", real_frame) 

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    real_gray = cv2.cvtColor(real_frame, cv2.COLOR_BGR2GRAY)

    if not bg_synced:
        bg_synced = is_frame_similar(real_gray, bg_gray, bg_synced)
        if bg_synced:
            print("Background synced with real-time camera.")
            print("Starting recording...")

    #append real frames to a list to start recording, stop when the background syncs again
    if bg_synced:
        bg_list.append(real_frame)
        index += 1
        if index >= Waitframes:
            bg_synced = is_frame_similar(real_gray, bg_gray, bg_synced)
            if not bg_synced:
                print("Recording Over.")
                break

#create a while loop to run background video from bg_list and real-time camera together
for i in range(len(bg_list)):
    ret_real, real_frame = real_cap.read()
    if not ret_real:
        break
    # Display the real-time camera frame
    cv2.imshow("Real-time Camera Frame", real_frame) 
    # Display the background frame from the list
    cv2.imshow("Background Frame", bg_list[i])
    # Check for 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    