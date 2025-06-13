import cv2
import numpy as np

rectangles = []  # List to store rectangles
points = []#list to store points clicked by the user

def get_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        b, g, r = frame[y, x]
        color_name = f"RGB: ({r}, {g}, {b})"
        print(color_name)

def click_event(event, x, y, flags, param):
        global points,rectangles      
        if event == cv2.EVENT_RBUTTONDOWN:
            print(f'Coordinate {len(points)}: ({x}, {y})')
            points.append((x, y))

            if len(points) == 2:
               rectangles.append(tuple(points))  # Store the rectangle defined by the two points
               points = []
               
#capture video
cap=cv2.VideoCapture(0) #Opens the webcam. The argument 0 selects the default camera.
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', click_event)

while True:
    _,frame=cap.read() 

    # Draw rectangles and their contours
    for pt1, pt2 in rectangles:
        # Draw rectangle
        #cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)  # Green rectangle[3][4]

        # Prepare contour points in order (top-left, top-right, bottom-right, bottom-left)
        contour = np.array([
            [pt1[0], pt1[1]],
            [pt2[0], pt1[1]],
            [pt2[0], pt2[1]],
            [pt1[0], pt2[1]]
        ]).reshape((-1, 1, 2))
        #cv2.drawContours(frame, [contour], 0, (255, 255, 255), 2)  # White contour[8]
        #create a mask for the rectangle
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        #fill the rectangle area in the mask
        cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill the rectangle area with white

        #calculate the mean color inside the rectangle
        mean_color = cv2.mean(frame, mask=mask)[:3]

        print(f"Mean color in rectangle: {mean_color}")
        # Show the frame with rectangles and contours
        
        # Define rectangle size and position (top-left corner)
        rect_top_left = (0, 0)
        rect_bottom_right = (50, 50)  # 50x50 rectangle at the top-left edge

        # Convert mean_color to integer tuple if needed
        mean_color_int = tuple(map(int, mean_color))

        # Draw filled rectangle at the edge with the mean color
        cv2.rectangle(frame, rect_top_left, rect_bottom_right, mean_color_int, thickness=-1)  # thickness=-1 fills the rectangle

    cv2.imshow("Frame", frame)
    key=cv2.waitKey(1) #wait for 1ms to press a key

    if key==13: #if key pressed is ascii 27, breaks away
        break

cap.release()
cv2.destroyAllWindows()
