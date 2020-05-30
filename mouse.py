import cv2
import numpy as np
import time

drawing = False
point1 = ()
point2 = ()
stop = 0

def mouse_drawing(event, x, y, flags, params):
    global point1, point2, drawing, stop
    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing is False:
            drawing = True
            point1 = (x, y)
            stop = 0
        else:
            drawing = False
            stop = 1

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing is True:
            point2 = (x, y)

cap = cv2.VideoCapture(0)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)


def get_image():
    global stop, point1, point2
    frame1 = []
    while(len(frame1) == 0):
        _, frame = cap.read()
        if point1 and point2:
            print(point1, " ", point2)
            cv2.rectangle(frame, point1, point2, (0, 255, 0))
            if(stop == 1):
                frame1 = np.copy(frame)
                frame1[point1[1]:point2[1], point1[0]:point2[0]] = 0
                stop = 0
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return frame1, point1, point2

while True:
    #get the two frames
    frame1, (x1, y1), (x2, y2) = get_image()
    point1 = (); point2 = ();
    frame2, (x2, y3), (x4, y4) = get_image()
    point1 = (); point2 = ();
    #Write your code here
    #stack the two frames
    numpy_horizontal = np.hstack((frame1, frame2))
    cv2.imshow("Frame", numpy_horizontal)
    #Stop the code until you press q
    while(cv2.waitKey() != ord('q')):
        continue
    
cap.release()
cv2.destroyAllWindows()
