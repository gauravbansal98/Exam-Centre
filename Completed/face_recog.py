from mtcnn.mtcnn import MTCNN
import face_recognition
import numpy as np
import tensorflow as tf
import cv2
import os

class Detector():
    def __init__(self):
        self.detector = MTCNN()
        self.detector.detect_faces(np.zeros((300, 300, 3)))
        self.session = tf.keras.backend.get_session()
        self.graph = tf.get_default_graph()
        # self.graph.finalize()

    def detect_faces(self, image):
        return self.detector.detect_faces(image)

def find_boxes(faces):
    boxes = []
    for result in faces:
        if result['confidence'] > .9:
            x, y, width, height = result['box']
            x_max = x + width
            y_max = y + height
            boxes.append((y, x+width, y+height, x))
    return boxes

def match_faces(filenames):
    return_dict = {}
    new_image = cv2.imread(os.path.join("static", filenames[-1]))
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    with detector.graph.as_default():
        faces = detector.detect_faces(new_image)
    boxes = find_boxes(faces)
    if(len(boxes) < 1):
        return_dict['msg'] = "Face not found"
        return return_dict
    encoding_to_match = face_recognition.face_encodings(new_image, boxes, num_jitters = 1)
    gt_encodings = []
    for image_name in filenames[:-1]:
        image = cv2.imread(os.path.join("static", image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        with detector.graph.as_default():
            faces = detector.detect_faces(image)
        boxes = find_boxes(faces)
        encodings = face_recognition.face_encodings(image, boxes, num_jitters = 1)
        if(len(encodings) > 0):
            gt_encodings.append(encodings[0])
    matches = face_recognition.compare_faces(gt_encodings, encoding_to_match[0], tolerance = .6)
    return_dict['count'] = float(sum(matches))
    return_dict['msg'] = "Face is found"
    return return_dict

def count_faces(image_name):
    return_dict = {}
    image = cv2.imread(os.path.join("static", image_name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    with detector.graph.as_default():
        faces = detector.detect_faces(image)
    length = 0
    confidences = []
    for face in faces:
        confidences.append((face['confidence']))
        length += 1
    return_dict['scores'] = confidences
    return_dict['msg'] = "Number of faces found " + str(length) 
    return return_dict


detector = Detector()