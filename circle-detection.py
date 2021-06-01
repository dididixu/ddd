# ## -*- coding: utf-8 -*-
import cv2
import numpy as np
import math

cv2.namedWindow("enhanced1", 0)
cv2.resizeWindow("enhanced1", 640, 480)
cv2.namedWindow("enhanced2", 0)
cv2.resizeWindow("enhanced2", 640, 480)


def polar(img):
    h, w = img.shape[:2]
    maxRadius = math.hypot(w / 2, h / 2)
    m = w / math.log(maxRadius)
    log_polar = cv2.logPolar(img, (w / 2, h / 2), m, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
    linear_polar = cv2.linearPolar(img, (w / 2, h / 2), maxRadius, cv2.WARP_FILL_OUTLIERS + cv2.INTER_LINEAR)
    cv2.imshow("enhanced1", log_polar)
    cv2.imshow("enhanced2", linear_polar)
    cv2.waitKey(0)


# 读取名称为 p10.png的图片
photo_path = r'stamp_photo/1_img1.jpg'
org = cv2.imread(photo_path, 1)
#
img = cv2.imread(photo_path, 1)
# polar(img)
img = cv2.medianBlur(img, 5)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 提取圆形
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 60, param1=170, param2=70, minRadius=50, maxRadius=200)
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=70, param2=70, minRadius=50, maxRadius=150)

circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    img_temp = img[i[0] - i[2]:i[0] + i[2], i[1] - i[2]:i[1] + i[2]]
    polar(img_temp)
    # draw the outer circle
    # cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    # cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

# 显示原图和处理后的图像
# cv2.imshow("enhanced1", org)
# cv2.imshow("enhanced2", img)

# cv2.waitKey(0)
