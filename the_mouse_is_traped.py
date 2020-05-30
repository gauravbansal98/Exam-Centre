import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from skimage import measure

drawing = False
point1 = ()
point2 = ()
stop = 0


def compare_images(imageA, imageB, ):
    # compute the structural similarity index for the images
    s = measure.compare_ssim(imageA, imageB)
    return s


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


def get_image():
    global stop, point1, point2
    frame1 = []
    while len(frame1) == 0:
        _, frame = cap.read()
        if point1 and point2:
            # print(point1, " ", point2)
            cv2.rectangle(frame, point1, point2, (0, 255, 0))
            if(stop == 1):
                frame1 = np.copy(frame)
                frame1[point1[1]:point2[1], point1[0]:point2[0]] = 0
                stop = 0
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return frame1, point1, point2


cap = cv2.VideoCapture(0)
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", mouse_drawing)
frame1, point11, point12 = get_image()
point1 = ()
point2 = ()
original = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
cv2.imshow('original', original)

while True:
    frame2, point21, point22 = get_image()
    point1 = ()
    point2 = ()
    frame2[point11[1]:point12[1], point11[0]:point12[0]] = 0
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Input', frame2)
    ssim = compare_images(original, frame2)
    print(ssim)
    if ssim < 0.7:
        print("ssim = ", ssim)
        print("the frame is changed!!!!")
        frame1, point11, point12 = get_image()
        point1 = ()
        point2 = ()
        original = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        cv2.imshow('original', original)

    c = cv2.waitKey(1)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()
