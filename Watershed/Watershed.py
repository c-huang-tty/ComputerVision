# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:25:08 2023

@author: tjhua
"""

#importing required libraries
import numpy as np
import cv2
# import matplotlib.pyplot as plt

# reading the image
image = cv2.imread("coins.png")
cv2.imshow('coins', image)

# converting image to grayscale format
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# apply thresholding
ret,thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
cv2.imshow('coins_thresh', thresh)

# get a kernel
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 2)
cv2.imshow('coins_opening', opening)

# extract the background from image
sure_bg = cv2.dilate(opening, kernel, iterations = 3)
cv2.imshow('coins_sure_bg', opening)

dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret,sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_bg)

# ret,markers = cv2.connectedComponents(sure_fg)

# markers = markers+1

# markers[unknown==255] = 0

# markers = cv2.watershed(image,markers)
# image[markers==-1] = [255,0,0]

# plt.imshow(sure_fg)

cv2.waitKey(0)
cv2.destroyAllWindows()