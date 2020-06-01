import numpy as np
import tensorflow as tf
import cv2
import os
from face_recog import detector
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator

sess = tf.Session() #Launch the graph in a session.
my_head_pose_estimator = CnnHeadPoseEstimator(sess) #Head pose estimation object

# Load the weights from the configuration folders
my_head_pose_estimator.load_yaw_variables(os.path.realpath("cnn_cccdd_30k.tf"))



def detect_pose(filename):
    image = cv2.imread(os.path.join("static", filename))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with detector.graph.as_default():
        faces = detector.detect_faces(rgb)
    if(len(faces) > 0):
        x, y, width, height = faces[0]['box']
        x -= width/4
        y -= height/5
        height += height/3
        width += width/3
        image = image[int(y):int(y+height), int(x):int(x+width), :]
        if(image.shape[0] == 0 or image.shape[1] == 0 or image.shape[2] == 0):
            return "Problem with image"
        image = cv2.resize(image, (150, 150))
        # Get the angles for roll, pitch and yaw
        yaw = my_head_pose_estimator.return_yaw(image)  # Evaluate the yaw angle using a CNN
        if(yaw[0][0][0] > -30 and yaw[0][0][0] < 40):
            return ("Looking Straight")
        elif(yaw[0][0][0] < -30 and yaw[0][0][0] > -55):
            return ("Looking left")
        elif(yaw[0][0][0] < -55):
            return ("Looking extreme left")
        elif(yaw[0][0][0] > 40 and yaw[0][0][0] < 65):
            return ("Looking right")
        elif(yaw[0][0][0] > 65):
            return ("Looking extreme right")
