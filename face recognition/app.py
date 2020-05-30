from flask import Flask
from flask import request
import os
import face_recognition
import cv2

app = Flask(__name__)

def match_faces(compare, filenames):
    new_image = cv2.imread(os.path.join("api", compare))
    boxes = face_recognition.face_locations(new_image,model="hog")
    encoding_to_match = face_recognition.face_encodings(new_image, boxes, num_jitters = 1)
    gt_encodings = []
    for image_name in filenames:
        image = cv2.imread(os.path.join("api", image_name)) 
        boxes = face_recognition.face_locations(image,model="hog")
        encodings = face_recognition.face_encodings(image, boxes, num_jitters = 1)
        gt_encodings.append(encodings[0])
    matches = face_recognition.compare_faces(gt_encodings, encoding_to_match[0], tolerance = .6)
    print(sum(matches))
    if(sum(matches) > 2):
        return "Matched"
    else:
        return "Does not match"

@app.route('/', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        file_1 = request.files['1']; file_1.save(os.path.join("api", file_1.filename))
        file_2 = request.files['2']; file_2.save(os.path.join("api", file_2.filename))
        file_3 = request.files['3']; file_3.save(os.path.join("api", file_3.filename))
        file_4 = request.files['4']; file_4.save(os.path.join("api", file_4.filename))
        file_5 = request.files['5']; file_5.save(os.path.join("api", file_5.filename))
        filenames = [file_1.filename, file_2.filename, file_3.filename, file_4.filename, file_5.filename]
        compare = request.files['compare']; compare.save(os.path.join("api", compare.filename))
        return match_faces(compare.filename, filenames)
    


if __name__ == '__main__':
    app.run()