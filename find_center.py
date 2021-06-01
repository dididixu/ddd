# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 12:11:24 2017

@author: zzz
"""
# import necessary packages

import imutils
import cv2
import matplotlib.pyplot as plt
from threshold import solve_stamp
# load the image,convert it to grayscale,blur it slightly
# and threshold it
image = cv2.imread('5.png')
img = solve_stamp(image)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # grayscale conversion
# blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Gaussian smoothing using a 5 x 5 kernel
# thresh = cv2.threshold(blurred, 250, 255, 0)[1]  # thresholding
thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# dst = 255 - thresh
# find contours in the thresholded image
binary, cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
temp = thresh
# cv2.imshow('12345',binary)
# return the set of outlines
# cnts = cnts[0] if imutils.is_cv2() else cnts[1]
# grap appropriate tuple value
# loop over the contours
for c in binary:
    # compute the center of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    print(cX, cY)
    # draw the contour and center of the shape on the image
    cv2.drawContours(image, binary, -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
    cv2.putText(image, "center", (cX - 20, cY - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(1, 2, 1)
plt.imshow(temp)
# plt.set_camp('binary')
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.show()
# cv2.imshow('123', image)
# cv2.imwrite('ahh.png', image)
# cv2.waitKey(0)
