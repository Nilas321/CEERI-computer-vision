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
count=0

def track_objects(contours, frame):
    global color_history, size_history,tracked_objects,detected_object,j,_id_counter,output_string
    objects = []
    #convert contours to list that can be tracked
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            (x, y, w, h) = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            objects.append((cx, cy, cnt))

    # If no objects detected, return empty tracked dict
    if not objects:
        return tracked_objects

    # Check for object tracking
    for obj_id, [prev_cx, prev_cy, _, stable_count,cnt] in tracked_objects.items():
        # Find nearest in new frame
        distances = [np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                      for (cx, cy, _) in objects]
        if distances:
            min_idx = np.argmin(distances)
            if distances[min_idx] < 20:  # Max allowed displacement
                new_cx, new_cy, new_cnt = objects[min_idx]
                tracked_objects[obj_id][0] = new_cx  # Update X
                tracked_objects[obj_id][1] = new_cy  # Update Y
                tracked_objects[obj_id][3] += 1      # Increment stability count
                tracked_objects[obj_id][4] = new_cnt  # Update contour
                
                del objects[min_idx]
  # Exit loop after updating this object
            else:
                tracked_objects[obj_id] = [prev_cx, prev_cy, False,-1,None]

    for obj_id in list(tracked_objects.keys()):
        if not tracked_objects[obj_id][2]:
            del tracked_objects[obj_id]  # Remove objects that are not found in current frame
    # Add new objects
    for cx, cy, cnt in objects:
        new_id = _id_counter
        _id_counter += 1
        tracked_objects[new_id] = [cx, cy, True, 0, cnt]  # Added contour storage

    for new_id, [cx, cy, _, stable_count,cnt] in tracked_objects.items():

        if stable_count > STABILITY_THRESHOLD:
            if cnt is None:
                    continue
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(hsv_frame, mask=mask)[:3]

            color_name = "Unknown"  # Default value

            h = mean_color[0]
            s = mean_color[1]
            v = mean_color[2]

            if (10 > h >= 0 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Red(Lower)"
            elif (179 > h >= 160 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Red(Upper)"
            elif (25 > h >= 10 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Orange"
            elif (35 >= h >= 25 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Yellow"
            elif (85 >= h >= 36 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Green"
            elif (100 >= h >= 86 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Cyan"
            elif (130 >= h >= 101 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Blue"
            elif (160 >= h >= 131 and 255 >= s >= 100 and 255 >= v >= 100):
                color_name = "Purple"
            elif (170 >= h >= 145 and 255 >= s >= 50 and 255 >= v >= 150):
                color_name = "Pink"
            elif (20 >= h >= 10 and 255 >= s >= 100 and 200 >= v >= 20):
                color_name = "Brown"
            elif (180 >= h >= 0 and 30 >= s >= 0 and 255 >= v >= 200):
                color_name = "White"
            elif (180 >= h >= 0 and 50 >= s >= 0 and 200 >= v >= 50):
                color_name = "Grey"
            elif (180 >= h >= 0 and 255 >= s >= 0 and 50 >= v >= 0):
                color_name = "Black"
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)

            # Draw rectangle and label
            (x, y, w, h) = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            (x, y, w, h) = cv2.boundingRect(cnt)

            #Let x be the variable for length across conveyer belt
            #x=27 to x=151 - Lane 1
            #x=152 to x=291 - Lane 2
            #x=292 to x=427 - Lane 3
            #x=428 to x=549 - Lane 4

            x1=215
            x2=383
            x3=551
            #y=35 be the line

            lane=[]
            x_end=x+w
            if x1>x>=48:
                if x1>x_end:
                    lane.append(1)
                    if y+h>=35>=y:
                        output_string="10000000"
                    else:
                        output_string="00000000"
                elif x2>x_end:
                    lane.append(1)
                    lane.append(2)
                    if y+h>=35>=y:
                        output_string="11000000"
                    else:
                        output_string="00000000"
                else:
                    lane.append(1)
                    lane.append(2)
                    lane.append(3)
                    if y+h>=35>=y:
                        output_string="11100000"
                    else:
                        output_string="00000000"
            elif x2>x:
                if x2>x_end:
                    lane.append(2)
                    if y+h>=35>=y:
                        output_string="01000000"
                    else:
                        output_string="00000000"
                else:
                    lane.append(2)
                    lane.append(3)
                    if y+h>=35>=y:
                        output_string="01100000"
                    else:
                        output_string="00000000"
            else:
                lane.append(3)
                if y+h>=35>=y:
                    output_string="00100000"
                else:
                    output_string="00000000"

            area = cv2.contourArea(cnt)

            if 500<area<2000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame,"Small " + "Lane: " + str(lane) + " Colour: " + color_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
            elif 2000<area<6000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame,"Medium " + "Lane: " + str(lane) + " Colour: " + color_name,(x,y -10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame,"Large " + "Lane: " + str(lane) + " Colour: " + color_name,(x,y -10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))

            #print(output_string)
    
        else:
            continue

    return tracked_objects

frame1 = cv2.imread("frame.jpg")
bg_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

bg_synced = False
bg_frames=[]

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # Use camera or video stream

print("Recording background... Press 'ENTER KEY' to stop.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    real_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    if not bg_synced:
        # Step 1: Check similarity between current real frame and reference background frame
        diff = cv2.absdiff(real_gray, bg_gray)
        score = np.sum(diff)
        print(score/100000)

        if score < 2500000:  #  Tune this threshold based on your setup
            print("Background synced!")
            bg_synced = True
            bg_frames.append(frame)
            time.sleep(0.75)
            continue
        else:
            cv2.putText(frame, "Syncing background...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Real Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # Keep checking next live frame

    
    cv2.imshow('Recording background', frame)

    if bg_synced:
        # Step 1: Check similarity between current real frame and reference background frame
        diff = cv2.absdiff(real_gray, bg_gray)
        score = np.sum(diff)
        print(score/100000)

        if score < 2500000:  #  Tune this threshold based on your setup
            print("Video Synced")
            bg_synced = True
            break
        else:
            cv2.putText(frame, "Syncing background...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Real Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # Keep checking next live frame

    # Open real-time camera
real_cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

bg_index = 0
bg_synced = False

while True:
    ret_real, real_frame = real_cap.read()
    if not ret_real:
        break

    real_gray = cv2.cvtColor(real_frame, cv2.COLOR_BGR2GRAY)

    if not bg_synced:
        # Step 1: Check similarity between current real frame and reference background frame
        diff = cv2.absdiff(real_gray, bg_gray)
        score = np.sum(diff)
        print(score/100000)

        if score < 2500000:  # ✅ Tune this threshold based on your setup
            print("Background synced!")
            bg_synced = True
            continue
        else:
            cv2.putText(real_frame, "Syncing background...", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Real Frame", real_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue  # Keep checking next live frame

while True:
    ret_real, real_frame = real_cap.read()
    if not ret_real:
        break

    real_gray = cv2.cvtColor(real_frame, cv2.COLOR_BGR2GRAY)

    for i in bg_frames:
        bg_gray=cv2.cvtColor(i,cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(real_gray, bg_gray)
        #gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        cv2.putText(real_frame,str(count) ,(50,100),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
        #apply gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(diff, (5, 5), 0)

        # Threshold the blurred image to create a binary mask
        _, foreground_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                valid_contours.append(cnt)
        # Track objects
        tracked_objects = track_objects(valid_contours, real_frame)
        #send_string_with_delay(output_string, 0.5)  # Simulate sending output string with delay


        # Display the original frame with detected foreground
        cv2.imshow('Foreground', real_frame)
        #cv2.imshow("Background", bg_frame)
        cv2.imshow('Foreground Mask', foreground_mask)
        count += 1
        key = cv2.waitKey(1)
        if key == 13:
            break
        #print("Tracked Objects:", tracked_objects])
        #print("Detected Objects:", detected_object)
        #print("Color History:", color_history)
        #print("Size History:", size_history)
        #print("\n\n")

real_cap.release()
cv2.destroyAllWindows()
