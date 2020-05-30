# it will take 2 images input API and return the value of SSI

from flask import Flask, request, jsonify
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import measure

app = Flask(__name__)


def compare_img(imageA, imageB):
    original = cv2.imread('resources/' + imageA)
    original = cv2.resize(original, (500, 400), interpolation=cv2.INTER_AREA)
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    input_img = cv2.imread('resources/' + imageB)
    input_img = cv2.resize(input_img, (500, 400), interpolation=cv2.INTER_AREA)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    s = measure.compare_ssim(original, input_img)
    ssim = str(s)
    return ssim


@app.route('/', methods=['GET', 'POST'])
def img_comp():
    if request.method == 'POST':
        file_1 = request.files['1']
        file_1.save(os.path.join('resources', file_1.filename))
        file_2 = request.files['2']
        file_2.save(os.path.join('resources', file_2.filename))
        # file.save(os.path.join('resources', file.filename))
        ssim = compare_img(file_1.filename, file_2.filename)
        return ssim
    else:
        return 'get request'


if __name__ == '__main__':
    app.run(debug=True)
