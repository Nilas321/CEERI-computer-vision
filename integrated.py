#!/usr/bin/python
# -*- coding:utf-8 -*-


"""
/*******************************************************************************
Copyright (c) 1983-2021 Advantech Co., Ltd.
********************************************************************************
THIS IS AN UNPUBLISHED WORK CONTAINING CONFIDENTIAL AND PROPRIETARY INFORMATION
WHICH IS THE PROPERTY OF ADVANTECH CORP., ANY DISCLOSURE, USE, OR REPRODUCTION,
WITHOUT WRITTEN AUTHORIZATION FROM ADVANTECH CORP., IS STRICTLY PROHIBITED.

================================================================================
REVISION HISTORY
--------------------------------------------------------------------------------
$Log:  $

--------------------------------------------------------------------------------
$NoKeywords:  $
*/

/******************************************************************************
*
* Windows Example:
*    StaticDO.py
*
* Example Category:
*    DIO
*
* Description:
*    This example demonstrates how to use Static DO function.
*
* Instructions for Running:
*    1. Set the 'deviceDescription' for opening the device.
*    2. Set the 'profilePath' to save the profile path of being initialized device.
*    3. Set the 'startPort'as the first port for Do .
*    4. Set the 'portCount'to decide how many sequential ports to operate Do.
*
* I/O Connections Overview:
*    Please refer to your hardware reference manual.
*
******************************************************************************/
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import random
import time
from Automation.BDaq import *
from Automation.BDaq.InstantDoCtrl import InstantDoCtrl
from Automation.BDaq.BDaqApi import AdxEnumToString, BioFailed

deviceDescription = "USB-5862,BID#0"
profilePath = u"C:\Advantech\DAQNavi\Examples\profile\DemoDevice.xml"
startPort = 0
portCount = 1

def AdvInstantDO(input):
    ret = ErrorCode.Success

    # Step 1: Create a instantDoCtrl for DO function.
    # Select a device by device number or device description and specify the access mode.
    # In this example we use ModeWrite mode so that we can fully control the device,
    # including configuring, sampling, etc.
    instantDoCtrl = InstantDoCtrl(deviceDescription)
    for _ in range(7):
        instantDoCtrl.loadProfile = profilePath
       
        # Step 2: Write DO ports
        dataBuffer = [0] * portCount
        for i in range(startPort, portCount + startPort):
            #inputVal is 255 for all On and 0 for all off 
            inputVal= input 
            
            dataBuffer[i-startPort] = inputVal
        time.sleep(2) # 2 milli-second delay
        ret = instantDoCtrl.writeAny(startPort, portCount, dataBuffer) # writes the output on the I/O board
        if BioFailed(ret):
            break
        print("DO output completed!")

    print("Port count supported:", instantDoCtrl.features.portCount)

    # Step 3: Close device and release any allocated resource.
    instantDoCtrl.dispose() 

    # If something wrong in this execution, print the error code on screen for tracking.
    if BioFailed(ret):
        enumStr = AdxEnumToString("ErrorCode", ret.value, 256)
        print("Some error occurred. And the last error code is %#x. [%s]" % (ret.value, enumStr))

    return 0

    


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
STABILITY_THRESHOLD = 3
# Read video
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)  # 0 for default camera
frame_count = 0

# Collect first 120 frames
background_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)
    frame_count += 1

# Compute median background
median_background = np.median(background_frames, axis=0).astype(np.uint8)

#display the median background
#cv2.imshow('Median Background', median_background)

def send_string_with_delay(string, delay_seconds):
        time.sleep(delay_seconds)  # Delay sending
        return string

#function to track objects
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

            lane=[]
            x_end=x+w
            if x1>x>=48:
                if x1>x_end:
                    lane.append(1)
                    input=16
                elif x2>x_end:
                    lane.append(1)
                    lane.append(2)
                    input=48
                else:
                    lane.append(1)
                    lane.append(2)
                    lane.append(3)
                    input=112
            elif x2>x:
                if x2>x_end:
                    lane.append(2)
                    input=32
                else:
                    lane.append(2)
                    lane.append(3)
                    input=96
            else:
                lane.append(3)
                input=64

            #y=35 be the line

            area = cv2.contourArea(cnt)

            if 500<area<2000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame,"Small " + "Lane: " + str(lane) + " Colour: " + color_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
                AdvInstantDO(input)
                
            elif 2000<area<6000:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame,"Medium " + "Lane: " + str(lane) + " Colour: " + color_name,(x,y -10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
                AdvInstantDO(input)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame,"Large " + "Lane: " + str(lane) + " Colour: " + color_name,(x,y -10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,0))
            print(output_string)
            
        else:
            continue

    return tracked_objects

while frame_count > 119:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute absolute difference between current frame and static background
    diff = cv2.absdiff(frame, median_background)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    #apply gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the blurred image to create a binary mask
    _, foreground_mask = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            valid_contours.append(cnt)
    # Track objects
    tracked_objects = track_objects(valid_contours, frame)
    #send_string_with_delay(output_string, 0.5)  # Simulate sending output string with delay


    # Display the original frame with detected foreground
    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    count = 0
    key = cv2.waitKey(1)
    if key == 13:
        break
    #print("Tracked Objects:", tracked_objects])
    #print("Detected Objects:", detected_object)
    print("Color History:", color_history)
    #print("Size History:", size_history)
    #print("\n\n")
    frame_count += 1
cap.release()
cv2.destroyAllWindows()