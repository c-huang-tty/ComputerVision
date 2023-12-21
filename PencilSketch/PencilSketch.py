# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 17:03:54 2023

@author: tjhua
"""

# https://towardsdatascience.com/generate-pencil-sketch-from-photo-in-python-7c56802d8acb

import cv2
import matplotlib.pyplot as plt

img = cv2.imread('chao_huang.jpg')
img = cv2.resize(img, (600, 800))

# show the image
# cv2.imshow('original image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# step 1: convert to gray image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# step 2: invert the gray image
invert_img = cv2.bitwise_not(gray_img)

# step 3: blur the inverted image
# The second argument to the function is the kernel size, if should be a pair of odd numbers.
# Larger the kernel size, more blurred the image will be and it will lose its subtle features.
blur_img = cv2.GaussianBlur(invert_img, (121, 121), 0)

# step 4: invert the blurred image
invert_blur_img = cv2.bitwise_not(blur_img)

# step 5: sketch
# The sketch can be obtained by performing bit-wise division between the grayscale image 
# and the inverted-blurred image.
sketch_img = cv2.divide(gray_img, invert_blur_img, scale = 256.0)

# step 6: save the image
cv2.imwrite('sketch.png', sketch_img)

# step 7: show the image
cv2.imshow('sketch image', sketch_img)
cv2.waitKey(0)
cv2.destroyAllWindows()