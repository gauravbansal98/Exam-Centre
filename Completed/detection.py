import numpy as np
import os
import tensorflow as tf

from collections import defaultdict

import label_map_util

import cv2

# cap = cv2.VideoCapture(0)

NUM_CLASSES = 90

PATH_TO_LABELS = 'mscoco_label_map.pbtxt'

PATH_TO_CKPT = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb'

class Detector():
    def __init__(self):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        # print(label_map)
        self.categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        # print(categories)
        self.category_index = label_map_util.create_category_index(self.categories)
        self.session = tf.keras.backend.get_session()
        self.graph = tf.get_default_graph()
        # with self.detection_graph.as_default():
        #     with tf.Session(graph=self.detection_graph) as self.sess:
        self.sess = tf.Session(graph=self.detection_graph)
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_objects(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores_array, classes, num_detections) = self.sess.run(
            [self.boxes, self.scores, self.classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        string = {}
        string["objects"] = []
        string["scores"] = []
        
        for scores in scores_array:
            for i, score in enumerate(scores):
                if(score  != 0):
                    string["objects"].append(self.category_index[i+1]['name'])
                    string["scores"].append(float(score))
        return string
        

def find_objects(image):
    image = cv2.imread(os.path.join("static", image))
    string = detector.detect_objects(image)
    return string
detector = Detector()