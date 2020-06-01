import cv2
import numpy as np
from skimage import measure
import os

def compare_img(filenames):
    print("i am here")
    original = cv2.imread(os.path.join("static", filenames[0]))
    original = cv2.resize(original, (500, 400), interpolation=cv2.INTER_AREA)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    input_img = cv2.imread(os.path.join("static", filenames[1]))
    input_img = cv2.resize(input_img, (500, 400), interpolation=cv2.INTER_AREA)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    s = measure.compare_ssim(original, input_img)
    if(s < .65):
        return "Image is changing"
    else:
        return "Image is not changing"