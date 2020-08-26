from flask import Flask, request
import os
import face_recog
import background_check
import head_pose
import vad
import requests
from flask import jsonify

import detection


app = Flask(__name__)
app.instance_path = "static"

@app.route('/')
def hello():
    return "hello"

def background(request):
    file_names = []
    string = {}
    for url in request.form:
        file_names.append(url+".jpg")
    for file in request.files:
        file_names.append(request.files[file].filename)
    if len(file_names) < 2:
        string['msg'] = "Number of files given is incorrect"
        return string
    file_names = sorted(file_names)
    string = background_check.compare_img(file_names[-2:])
    return string

def match(request):
    file_names = []
    string = {}
    for url in request.form:
        file_names.append(url+".jpg")
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".jpg", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_names.append(request.files[file].filename)
        request.files[file].save( os.path.join("static", request.files[file].filename))
    if len(file_names) == 0:
        string['msg'] = "No file is found"
        return string
    file_names = sorted(file_names)
    string = face_recog.match_faces(file_names)
    return string

def no_faces(request):
    file_names = []
    string = {}
    for url in request.form:
        file_names.append(url+".jpg")
    for file in request.files:
        file_names.append(request.files[file].filename)
    if(file_names == []):
        string['msg'] = "No file is found"
        return string
    file_names = sorted(file_names)
    string = face_recog.count_faces(file_names[-1])
    return string

def estimation(request):
    file_names = []
    string = {}
    for url in request.form:
        file_names.append(url+".jpg")
    for file in request.files:
        file_names.append(request.files[file].filename)
    if(file_names == []):
        string['msg'] = "No file is found" 
        return string
    file_names = sorted(file_names)
    string = head_pose.detect_pose(file_names[-1])
    return string

@app.route('/all', methods=['POST'])
def all():
    file_names = []
    for url in request.form:
        file_names.append(url+".jpg")
    for file in request.files:
        file_names.append(request.files[file].filename)
    string = {}
    string['face_match'] = match(request)
    string['head_pose_estimation'] = estimation(request)
    string['no_of_faces'] = no_faces(request)
    string['background_match'] = background(request)
    [os.remove(os.path.join("static", filename)) for filename in file_names]
    return jsonify(string)


@app.route('/background_check', methods=['POST'])   
def compare_background():
    file_names = []
    string = {}
    for url in request.form:
        file_names.append(url+".jpg")
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".jpg", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_names.append(request.files[file].filename)
        request.files[file].save( os.path.join("static", request.files[file].filename))
    if len(file_names) != 2:
        string['msg'] = "Number of files given is incorrect"
        return jsonify(string)
    string = background_check.compare_img(file_names)
    [os.remove(os.path.join("static", filename)) for filename in file_names]
    return jsonify(string)
    

@app.route('/face_match', methods=['POST'])
def face_match():
    file_names = []
    string = {}
    for url in request.form:
        file_names.append(url+".jpg")
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".jpg", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_names.append(request.files[file].filename)
        request.files[file].save( os.path.join("static", request.files[file].filename))
    if len(file_names) == 0:
        string['msg'] = "No file is found"
        return jsonify(string)
    file_names = sorted(file_names)
    string = face_recog.match_faces(file_names)
    [os.remove(os.path.join("static", filename)) for filename in file_names]
    return jsonify(string)
    
@app.route('/count_faces', methods=['POST'])
def no_of_faces():
    string = {}
    file_name = ""
    for url in request.form:
        file_name = url+".jpg"
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".jpg", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_name = request.files[file].filename
        request.files[file].save(os.path.join("static", request.files[file].filename))
    if(file_name == ""):
        string['msg'] = "No file is found"
        return jsonify(string)
    string = face_recog.count_faces(file_name)
    os.remove(os.path.join("static", file_name))
    return jsonify(string)

@app.route('/detect_objects', methods=['POST'])
def detect_objects():
    string = {}
    file_name = ""
    for url in request.form:
        file_name = url+".jpg"
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".jpg", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_name = request.files[file].filename
        request.files[file].save(os.path.join("static", request.files[file].filename))
    if(file_name == ""):
        string['msg'] = "No file is found"
        return jsonify(string)
    string = detection.find_objects(file_name)
    os.remove(os.path.join("static", file_name))
    return jsonify(string)

@app.route('/voice_detect', methods=['POST'])
def voice_detect():
    file_name = ""
    string = {}
    for url in request.form:
        file_name = url+".wav"
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".wav", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_name = request.files[file].filename
        request.files[file].save(os.path.join("static", request.files[file].filename)) 
    if(file_name == ""):
        string['msg'] = "No file is found"
        return jsonify(string)   
    string = vad.detect_audio(file_name)
    os.remove(os.path.join("static", file_name))
    return jsonify(string)

@app.route('/head_pose', methods=['POST'])
def head_pose_estimation():
    file_name = ""
    string = {}
    for url in request.form:
        file_name = url+".jpg"
        r = requests.get(request.form[url])
        with app.open_instance_resource(url+".jpg", 'wb') as f:
            f.write(r.content)
    for file in request.files:
        file_name = request.files[file].filename
        request.files[file].save(os.path.join("static", request.files[file].filename))
    if(file_name == ""):
        string['msg'] = "No file is found" 
        return jsonify(string)
    string = head_pose.detect_pose(file_name)
    os.remove(os.path.join("static", file_name))
    return jsonify(string)


if __name__ == '__main__':
    app.run()
