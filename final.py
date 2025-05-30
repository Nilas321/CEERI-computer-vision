import cv2
import numpy as np

count = 0

# Read video
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0 for default camera
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)  )

# Collect first 120 frames
background_frames = []
for _ in range(120):
    ret, frame = cap.read()
    if not ret:
        break
    background_frames.append(frame)

# Compute median background
median_background = np.median(background_frames, axis=0).astype(np.uint8)

#display the median background
cv2.imshow('Median Background', median_background)

while cap.isOpened():
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

    for cnt in valid_contours:

        #Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a mask for the current contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)  # Fill the contour area with white
        
        # Calculate the mean color within the contour area of the ROI
        mean_color = cv2.mean(hsv_frame, mask=mask)[:3]
        #print(f"Mean color in rectangle {count}: {mean_color}")
        h=mean_color[0]
        s=mean_color[1]
        v=mean_color[2]
        if (10>h>=0 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Red(Lower)")
        elif (179>h>=160 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Red(Upper)")
        elif (25>h>=10 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Orange")
        elif (35>=h>=25 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Yellow")
        elif (85>=h>=36 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Green")
        elif (100>=h>=86 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Cyan")
        elif (130>=h>=101 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Blue")
        elif (160>=h>=131 and 255>=s>=100 and 255>=v>=100):
            print(f"Mean Colour in the rectange {count}: Purple")
        elif (170>=h>=145 and 255>=s>=50 and 255>=v>=150):
            print(f"Mean Colour in the rectange {count}: Pink")
        elif (20>=h>=10 and 255>=s>=100 and 200>=v>=20):
            print(f"Mean Colour in the rectange {count}: Brown")
        elif (180>=h>=0 and 30>=s>=0 and 255>=v>=200):
            print(f"Mean Colour in the rectange {count}: White")
        elif (180>=h>=0 and 50>=s>=0 and 200>=v>=50):
            print(f"Mean Colour in the rectange {count}: Grey")
        elif (180>=h>=0 and 255>=s>=0 and 50>=v>=0):
            print(f"Mean Colour in the rectange {count}: Black")
        
        # Draw a filled rectangle (50x50) at the top-left corner of the frame,
        # colored with the mean color found in the contour area
        rect_top_left = (0, 0 + 50 * count)
        rect_bottom_right = (50, 50 + 50 * count)
        cv2.putText(frame,"box number" + str(count),(55,50 + 50 * count),1,1,(255,255,0))
        count += 1
        
        mean_color_int = tuple(map(int, mean_color))
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, mean_color_int, thickness=-1)
        
    #for cnt in valid_contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 500<area<1000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame,"Small,box number" + str(count),(x,y),1,1,(255,255,0))
        elif 1000<area<2000:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame,"Medium,box number" + str(count),(x,y),1,1,(255,255,0))
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame,"Large,box number" + str(count),(x,y),1,1,(255,255,0))
    # Draw a green bounding rectangle around the detected contour in the ROI
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    

    # Display the original frame with detected foreground
    cv2.imshow('Foreground', frame)
    cv2.imshow('Foreground Mask', foreground_mask)
    count = 0
    key = cv2.waitKey(1)
    if key == 13:
        break
cap.release()
cv2.destroyAllWindows()
