 # -*- coding: utf-8 -*-
"""
File: face_detection_mtcnn.py
Created on Sat May 30 2020

@author: Ankit Bansal

=========================================================================
Write a summary of what this file does
=========================================================================
"""

# face detection with mtcnn on a photograph
import stable_hopenetlite
import utils

import cv2
import time
from PIL import Image
from matplotlib import pyplot

from mtcnn.mtcnn import MTCNN
import torch
import torch.nn.functional as F
from torchvision import transforms

out_window = 'Detections'

# draw an image with detected objects
def draw_image_with_boxes(image, result_list):
	# plot the image
    image = image.copy()

	# plot each box
    for result in result_list:
        # get coordinates
        x, y, width, height = result['box']
        # draw the box
        cv2.rectangle(image, (x,y), (x+width, y+height), (255,0,0), 2)

        # draw the dots
        for key, value in result['keypoints'].items():
            # create and draw dot
            cv2.circle(image, value, 2, (0,255,0))

    return image
    
# draw each face separately
def draw_faces(image, result_list):
    # extract and plot each detected face in a photograph
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(image[y1:y2, x1:x2])
	# show the plot
	pyplot.show()

# create the detector, using default weights
detector = MTCNN()

# filename = r'images\test2.jpg'

# # load image from file
# pixels = cv2.imread(filename)
# pixels_RGB = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

# # detect faces in the image
# faces = detector.detect_faces(pixels_RGB)
# # display faces on the original image
# draw_image_with_boxes(pixels, faces)
# cv2.waitKey(1)

# display faces from the original image
# draw_faces(pixels_RGB, faces)

pos_net = stable_hopenetlite.shufflenet_v2_x1_0()
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")
pos_net.load_state_dict(saved_state_dict, strict=False)
pos_net.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)


idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor)

transformations = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


while True:
    start = time.time()
    ret, frame = cap.read()
    if not ret:
        print('Done!')
        break;
    
    frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(frame_RGB)
    image = draw_image_with_boxes(frame.copy(), faces)
    fps = 1/ (time.time()-start)
    cv2.putText(image, 'FPS: {:.3f}'.format(fps), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
    cv2.imshow(out_window, image)
    for face in faces:
        x_min, y_min, width, height = face['box']
        x_min -= width/2
        y_min -= height/2
        x_max = x_min + width*2
        y_max = y_min + height*2

        x_min = int(max(x_min, 0));
        y_min = int(max(y_min, 0))
        x_max = int(min(frame.shape[1], x_max))
        y_max = int(min(frame.shape[0], y_max))

        # Crop image
        img = frame_RGB[y_min:y_max,x_min:x_max]
        # cv2.imshow("dfgfdg", img)

        img = Image.fromarray(img)
        
        # Transform
        img = transformations(img)
        img_shape = img.size()
        # img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img.unsqueeze_(0)

        y, p, r = pos_net.forward(img)

        yaw_predicted = F.softmax(y)
        pitch_predicted = F.softmax(p)
        roll_predicted = F.softmax(r)

        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
        utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = height/2)
        orientation = "yaw {:.3}, pitch {:.3}, roll {:.3}".format(yaw_predicted, pitch_predicted, roll_predicted)
        print(orientation)
        cv2.putText(frame, orientation, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
    
    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

