from mtcnn.mtcnn import MTCNN
import cv2

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

detector = MTCNN()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_RGB)
        image = draw_image_with_boxes(frame.copy(), faces)
        cv2.imshow('image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
