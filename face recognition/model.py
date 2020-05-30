from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
import time
from mtcnn.mtcnn import MTCNN

# images_ankit = ["2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg"]
# images_gaurav = ["7.jpg", "8.jpg", "9.jpg", "10.jpg", "11.jpg"]
# for i in images_gaurav:
# 	rgb = cv2.imread(i)
# 	# rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# 	boxes = face_recognition.face_locations(rgb,
#     model="hog")
# 	img = rgb[boxes[0][0]:boxes[0][2], boxes[0][3]:boxes[0][1], :]
# 	cv2.imwrite(os.path.join("gaurav", i), img)

for image in os.listdir('.'):
	if(len(image.split('.')) > 1 and image.split('.')[1] == 'jpg'):
		print("Analyzing image ", image)
		rgb = cv2.imread(image)
		a = time.time()
		boxes = face_recognition.face_locations(rgb,model="cnn")
		print(boxes)
		print(time.time()-a)
		encoding_match = face_recognition.face_encodings(rgb, boxes, num_jitters = 1)
		for dir in os.listdir('.'):
			encodings = []
			if(len(dir.split('.')) == 1):
				print("\tAnalyzing dir ", dir)
				for img in os.listdir(dir):
					rgb = cv2.imread(os.path.join(dir, img))
					boxes = face_recognition.face_locations(rgb,model="hog")
					encoding = face_recognition.face_encodings(rgb, boxes, num_jitters = 1)
					if(len(encoding) > 0):
						encodings.append(encoding[0])
				if(len(encoding_match) > 0):
					print("\t\t", face_recognition.compare_faces(encodings, encoding_match[0], tolerance = .6))

# compute the facial embedding for the face
# encodings = face_recognition.face_encodings(rgb, boxes)
# print((encodings[0]))

# for image in os.listdir('.'):
#     print(image)
#     img = cv2.imread(image)
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     boxes = face_recognition.face_locations(rgb,
#     model="hog")

#     # compute the facial embedding for the face
#     encoding = face_recognition.face_encodings(rgb, boxes)
#     print(face_recognition.face_distance(encodings, encoding[0]))
#     print(cosine_similarity(encodings, encoding))
'''
# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
	# extract the person name from the image path
	print("[INFO] processing image {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# load the input image and convert it from RGB (OpenCV ordering)
	# to dlib ordering (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# detect the (x, y)-coordinates of the bounding boxes
	# corresponding to each face in the input image
	boxes = face_recognition.face_locations(rgb,
		model=args["detection_method"])

	# compute the facial embedding for the face
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop over the encodings
	for encoding in encodings:
		# add each encoding + name to our set of known names and
		# encodings
		knownEncodings.append(encoding)
		knownNames.append(name)

# dump the facial encodings + names to disk
print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(args["encodings"], "wb")
f.write(pickle.dumps(data))
f.close()
'''