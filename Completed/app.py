from flask import Flask, request
import os
import face_recog
import background_check
import head_pose
import vad


app = Flask(__name__)

@app.route('/')
def hello():
    return "hello"


@app.route('/background_check', methods=['POST'])   
def compare_background():
    files = ['1', '2']
    [request.files[file].save(os.path.join("static", request.files[file].filename)) for file in files]
    filenames = [request.files[file].filename for file in files]
    print(filenames)
    string = background_check.compare_img(filenames)
    [os.remove(os.path.join("static", filename)) for filename in filenames]
    return string

@app.route('/face_match', methods=['POST'])
def face_match():
    files = ['compare', '1', '2', '3', '4', '5']
    [request.files[file].save(os.path.join("static", request.files[file].filename)) for file in files]
    filenames = [request.files[file].filename for file in files]
    string = face_recog.match_faces(filenames)
    [os.remove(os.path.join("static", filename)) for filename in filenames]
    return string
    
@app.route('/count_faces', methods=['POST'])
def no_of_faces():
    request.files['1'].save(os.path.join("static", request.files['1'].filename))
    string = face_recog.count_faces(request.files['1'].filename)
    os.remove(os.path.join("static", request.files['1'].filename))
    return string

@app.route('/voice_detect', methods=['POST'])
def voice_detect():
    request.files['audio'].save(os.path.join("static", request.files['audio'].filename))
    string = vad.detect_audio(request.files['audio'].filename)
    os.remove(os.path.join("static", request.files['audio'].filename))
    return string

@app.route('/head_pose', methods=['POST'])
def head_pose_estimation():
    request.files['1'].save(os.path.join("static", request.files['1'].filename))
    string = head_pose.detect_pose(request.files["1"].filename)
    os.remove(os.path.join("static", request.files['1'].filename))
    return string


if __name__ == '__main__':
    app.run()