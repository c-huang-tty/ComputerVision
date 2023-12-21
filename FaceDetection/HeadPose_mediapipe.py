# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 10:32:01 2023

@author: tjhua
"""

# https://medium.com/towards-data-science/head-pose-estimation-using-python-d165d3541600

import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0) 

while(1):
    ret, image = cap.read()
    
    # On the OpenCV library, the image is in BGR color space. 
    # Meanwhile, the mediapipe library needs an image with RGB color space.
    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    
   # To improve performance
    image.flags.writeable = False

    # Get the results
    # The mediapipe’s face landmark detection algorithm catches around 468 keypoints 
    # from a face. Each keypoint is on 3D coordinates.
    results = face_mesh.process(image)
    
    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            ### Retrieve the 2D and the 3D coordinates
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
            
            # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)

            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)

            ### Get the camera matrix
            focal_length = 1 * img_w

            cam_matrix = np.array([ [focal_length, 0,            img_h / 2],
                                    [0,            focal_length, img_w / 2],
                                    [0,            0,            1]])

            # The Distance Matrix (Assume no camera distortion)
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            # Solve PnP
            #  cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs) -> retval, rvec, tvec
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            # Get rotational matrix
            # https://medium.com/@sim30217/rotation-vector-vs-rotation-matrices-2b7ab7287b47
            rmat, jac = cv2.Rodrigues(rot_vec)

            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            # Get the y rotation degree
            x = angles[0] * 360 - 8 # calibration according to the position of the camera
            y = angles[1] * 360

            print('x = ', x, 'y = ', y)

            # See where the user's head tilting
            if y < -5:
                text = "Looking Left"
            elif y > 5:
                text = "Looking Right"
            elif x < -5:
                text = "Looking Down"
            elif x > 5:
                text = "Looking Up"
            else:
                text = "Forward"

            # Display the nose direction
            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
            
            for p in face_2d:
                cv2.circle(image, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
                
            cv2.line(image, p1, p2, (255, 0, 0), 2)

            # Add the text on the image
            cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Head Pose Estimation', image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
