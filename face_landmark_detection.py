import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

#img_dir = input("Please enter your image directory: ")
pred_dir = "cs445-final/utils/shape_predictor_68_face_landmarks.dat"

def facial_landmarks_detection(img_dir):
    # Read in image, resize it and convert it to gray scale
    image = cv2.imread(img_dir)
    image = imutils.resize(image, width = 500)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Load the dlib face detector and predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(pred_dir)

    # Locate the face in the given gray_scale image
    rects = detector(image_gray, 1)

    # Make a list to store data points
    points = np.zeros((68,2))

    for i, rect in enumerate(rects):
        # Get each certain facial landmark
        shape = predictor(image_gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Convert the rectangle to openCV bounding box
        x, y, width, height = face_utils.rect_to_bb(rect)
        start = (x, y)
        end = (width + x, height + y)
        color = (255, 0, 0)
        cv2.rectangle(image, start, end, color, 2)

        # Plot the point on each key landmark position
        i = 0
        for x, y in shape:
            point_color = (0, 255, 0)
            cv2.circle(image, (x, y), 2, point_color, 2)
            points[i][0] = x
            points[i][1] = y
            i += 1

    #plt.figure()
    #plt.imshow(image)
    return points