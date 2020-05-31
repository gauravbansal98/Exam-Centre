import cv2
import os

for dir in os.listdir('.'):
    if(dir.split('.')[1] != 'zip'):
        print("Selecting in ", dir)
        for image in os.listdir(dir):
            print(image)