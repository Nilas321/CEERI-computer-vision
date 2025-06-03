import time
import cv2
import numpy as np

#capture video
cap=cv2.VideoCapture(0,cv2.CAP_DSHOW) #Opens the webcam. The argument 0 selects the default camera.

while True:
    _,frame=cap.read() #capture frames from a video source_Returns a tuple containing two values
                       #boolean(indicating whether the frame captured successfully)_frame(NumPy array representing the captured image)'''
                       #_ contains the boolean value, frame contains the image array

    req_area=frame[194:438,210:590]
    #x1=210 , y1=194     top-left point
    #x2=590 , y2=438     bottom-right point

    gray_frame=cv2.cvtColor(req_area,cv2.COLOR_RGB2GRAY)

    _,threshold=cv2.threshold(gray_frame,200,255,cv2.THRESH_BINARY)
    #threshold=cv2.bitwise_not(threshold)
    def send_string_with_delay(string, delay_seconds):
        time.sleep(delay_seconds)  # Delay sending
        return string  
    #detect objects
    contours,_=cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        mask = np.zeros(req_area.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # Fill the rectangle area with white
        mean_color = cv2.mean(req_area, mask=mask)[:3]
        #print(f"Mean color in rectangle: {mean_color}")
         # Define rectangle size and position (top-left corner)
        rect_top_left = (0, 0)
        rect_bottom_right = (50, 50)  # 50x50 rectangle at the top-left edge
        mean_color_int = tuple(map(int, mean_color))
        # Draw filled rectangle at the edge with the mean color
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, mean_color_int, thickness=-1)  # thickness=-1 fills the rectangle
        
        (x,y,w,h)=cv2.boundingRect(cnt)
        cv2.rectangle(req_area,(x,y),(x+w,y+h),(0,255,0),2)
    
    #show video
    cv2.imshow("Frame", frame)
    #cv2.imshow("Required Area", req_area)
    #cv2.imshow("Binarized Frame", threshold)

    def click_event(event, x, y, flags, param):      # THIS IS JUST TO GET THE CO-ORDINATE SOF THE REQUIRED AREA!
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f'Coordinates: ({x}, {y})')
        # You can also draw a circle or mark the point on the image, if needed
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1) # Mark the clicked point
    cv2.imshow('image', frame)
    cv2.setMouseCallback('image', click_event)

    key=cv2.waitKey(1) #wait for 1ms to press a key

    if key==13: #if key pressed is ascii 27, breaks away
        break

cap.release()
cv2.destroyAllWindows()
