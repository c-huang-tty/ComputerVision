# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:15:48 2023

@author: tjhua
"""
# https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

# import cv2
# import numpy as np
# modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
# configFile = "models/deploy.prototxt.txt"
# net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import math
import dlib

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Read Image
cap = cv2.VideoCapture(0)
ret, img = cap.read()
size = img.shape
font = cv2.FONT_HERSHEY_SIMPLEX 
    
#2D image points. 

# 3D model points.
model_points = np.array([
                            (   0.0,    0.0,    0.0),             # Nose tip
                            (   0.0, -330.0,  -65.0),        # Chin
                            (-225.0,  170.0, -135.0),     # Left eye left corner
                            ( 225.0,  170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            ( 150.0, -150.0, -125.0)      # Right mouth corner
                        
                        ])


# Camera internals

focal_length = size[1]
center = (size[1]/2, size[0]/2)
camera_matrix = np.array([[focal_length, 0,            center[0]],
                          [0,            focal_length, center[1]],
                          [0,            0,            1]], 
                          dtype = "double")

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
    
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
        
	# return the list of (x, y)-coordinates
	return coords

def draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix, color=(255, 255, 0), line_width=2):
    """Draw a 3D box as annotation of pose"""
    point_3d = []
    dist_coeffs = np.zeros((4,1))
    rear_size = 1
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size,  rear_size, rear_depth))
    point_3d.append(( rear_size,  rear_size, rear_depth))
    point_3d.append(( rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = img.shape[1]
    front_depth = front_size*2
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size,  front_size, front_depth))
    point_3d.append(( front_size,  front_size, front_depth))
    point_3d.append(( front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=float).reshape(-1, 3)

    # Map to 2d img points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeffs)
    point_2d = np.int32(point_2d.reshape(-1, 2))
    

    # # Draw all the lines
    # cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    k = (point_2d[5] + point_2d[8])//2
    # cv2.line(img, tuple(point_2d[1]), tuple(
    #     point_2d[6]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[2]), tuple(
    #     point_2d[7]), color, line_width, cv2.LINE_AA)
    # cv2.line(img, tuple(point_2d[3]), tuple(
    #     point_2d[8]), color, line_width, cv2.LINE_AA)
    
    return(point_2d[2], k)

while(1):
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if ret == True:
        faceboxes = detector(gray_img)
        
        for facebox in faceboxes:
            # face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
            landmarks = predictor(image=gray_img, box=facebox)
            landmarks = shape_to_np(landmarks)
            
            # mark_detector.draw_marks(img, marks, color=(0, 255, 0))
            image_points = np.array([landmarks[30],     # Nose tip
                                     landmarks[8],      # Chin
                                     landmarks[36],     # Left eye left corner
                                     landmarks[45],     # Right eye right corne
                                     landmarks[48],     # Left Mouth corner
                                     landmarks[54]      # Right mouth corner
                                    ], dtype="double")
            dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
            
            # print("Rotation Vector:\n {0}".format(rotation_vector))
            # print("Translation Vector:\n {0}".format(translation_vector))
            
            # Project a 3D point (0, 0, 1000.0) onto the image plane.
            # We use this to draw a line sticking out of the nose
            
            (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
            
            for p in image_points:
                cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
            
            
            p1 = ( int(image_points[0][0]), int(image_points[0][1]))
            p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
            x1, x2 = draw_annotation_box(img, rotation_vector, translation_vector, camera_matrix)

            cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.line(img, tuple(x1), tuple(x2), (255, 255, 0), 2)
            
            # for (x, y) in shape:
            #     cv2.circle(img, (x, y), 4, (255, 255, 0), -1)
            # cv2.putText(img, str(p1), p1, font, 1, (0, 255, 255), 1)
            
            try:
                m = (p2[1] - p1[1])/(p2[0] - p1[0])
                ang1 = int(math.degrees(math.atan(m)))
            except:
                ang1 = 90
                
            try:
                m = (x2[1] - x1[1])/(x2[0] - x1[0])
                ang2 = int(math.degrees(math.atan(-1/m)))
            except:
                ang2 = 90
                
                # print('div by zero error')
            cv2.putText(img, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
            cv2.putText(img, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
            
        cv2.imshow('img', img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cv2.destroyAllWindows()
cap.release()
