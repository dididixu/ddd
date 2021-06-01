# coding=utf-8
# Enhancing the contrast加强对比度
import cv2
import matplotlib.pyplot as plt

# gray image
img = cv2.imread('11.jpg', cv2.IMREAD_GRAYSCALE)
hist_cv = cv2.calcHist([img], [0], None, [256], [0, 256])
plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.plot(hist_cv)
# equalize the histogram of the input image
histeq = cv2.equalizeHist(img)
hist_cv1 = cv2.calcHist([histeq], [0], None, [256], [0, 256])
plt.subplot(223), plt.plot(hist_cv1)
plt.show()
cv2.imshow('Input', img)
cv2.imshow('Histogram equalized', histeq)
cv2.waitKey(0)
# color image
img = cv2.imread('11.jpg', cv2.IMREAD_ANYCOLOR)
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
# equalize the histogram of the Y channel
img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
# convert the YUV image back to RGB format
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Color input image', img)
cv2.imshow('Histogram equalized', img_output)
cv2.imwrite('test4.jpg', img_output)
cv2.waitKey(0)
