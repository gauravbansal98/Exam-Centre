from PIL import Image
import stable_hopenetlite
import cv2
import torch
import dlib
import utils
import torch.nn.functional as F
from torchvision import transforms
import time

pos_net = stable_hopenetlite.shufflenet_v2_x1_0()
saved_state_dict = torch.load('model/shuff_epoch_120.pkl', map_location="cpu")
pos_net.load_state_dict(saved_state_dict, strict=False)
pos_net.eval()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Unable to connect to camera.")
    exit(-1)

detector = dlib.get_frontal_face_detector()
#detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

idx_tensor = [idx for idx in range(66)]
idx_tensor = torch.FloatTensor(idx_tensor)

transformations = transforms.Compose([transforms.Scale(224),
transforms.CenterCrop(224), transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

while cap.isOpened():
    ret, frame = cap.read()
    cv2_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    if ret:
        start = time.time()
        face_rects = detector(cv2_frame, 1)
        print("time taken to detect ", time.time()-start)
        # for face in face_rects:
        if(len(face_rects)) > 0:
            x_min = face_rects[0].left()
            y_min = face_rects[0].top()
            x_max = face_rects[0].right()
            y_max = face_rects[0].bottom()
            # x_min = face.rect.left()
            # y_min = face.rect.top()
            # x_max = face.rect.right()
            # y_max = face.rect.bottom()
            bbox_width = abs(x_max - x_min)
            bbox_height = abs(y_max - y_min)
            x_min -= 2 * bbox_width / 4
            x_max += 2 * bbox_width / 4
            y_min -= 3 * bbox_height / 4
            y_max += bbox_height / 4
            x_min = int(max(x_min, 0)); y_min = int(max(y_min, 0))
            x_max = int(min(frame.shape[1], x_max)); y_max = int(min(frame.shape[0], y_max))
            # Crop image
            img = cv2_frame[y_min:y_max,x_min:x_max]
            img = Image.fromarray(img)

            # Transform
            img = transformations(img)
            img_shape = img.size()
            img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
            
            y, p, r = pos_net.forward(torch.Tensor(img))
            yaw_predicted = F.softmax(y)
            pitch_predicted = F.softmax(p)
            roll_predicted = F.softmax(r)
            # Get continuous predictions in degrees.
            yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
            pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
            roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99
            utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
            print("yaw {}, pitch {}, roll {}".format(yaw_predicted, pitch_predicted, roll_predicted))
        
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
