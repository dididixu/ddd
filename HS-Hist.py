# coding=utf-8
import matplotlib.pyplot as plt
import cv2
import numpy as np
from test_photo import solve

np.set_printoptions(threshold=np.inf)
image = cv2.imread('test4.jpg')
# red_mask=cv2.inRange(hsv,np.array([156,43,46]),np.array([180,255,255]))
hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
low_range = np.array([0, 10, 10])
high_range = np.array([60, 255, 255])
th = cv2.inRange(hue_image, low_range, high_range)
red_mask = cv2.inRange(hue_image, np.array([155, 10, 10]), np.array([180, 255, 255]))
# print(th)
index1 = th == 255
img = np.zeros(image.shape, np.uint8)
img[:, :] = (255, 255, 255)
img[index1] = image[index1]  # (0,0,255)

index2 = red_mask == 255
img_red = np.zeros(image.shape, np.uint8)
img_red[:, :] = (255, 255, 255)
img_red[index2] = image[index2]  # (0,0,255)

index3 = index1 | index2
img_sum = np.zeros(image.shape, np.uint8)
img_sum[:, :] = (255, 255, 255)
img_sum[index3] = image[index3]  # (0,0,255)

# cv2.imshow('img', img)
cv2.imwrite('test.jpg', img)
# cv2.imshow('img', img)
cv2.imwrite('test2.jpg', img_red)
cv2.imwrite('test3.jpg', img_sum)
# # hist = cv.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
# hist1 = cv2.calcHist([hue_image], [0], None, [256], [0, 255])
# hist2 = cv2.calcHist([hue_image], [1], None, [256], [0, 255])
# hist3 = cv2.calcHist([hue_image], [2], None, [256], [0, 255])
# plt.figure(1)
# plt.plot(hist1)
# plt.figure(2)
# plt.plot(hist2)
# plt.figure(3)
# plt.plot(hist3)
# plt.show()

image1, image2, image3, image4 = solve('test3.jpg')
plt.subplot(2, 2, 1), plt.imshow(image1, 'gray')  # 默认彩色，另一种彩色bgr
plt.subplot(2, 2, 2), plt.imshow(image2, 'gray')
plt.subplot(2, 2, 3), plt.imshow(image3, 'gray')
plt.subplot(2, 2, 4), plt.imshow(image4, 'gray')
cv2.imwrite(r"C:\Users\49429\Desktop\1_result.jpg",image4)
plt.show()

cv2.waitKey(0)
