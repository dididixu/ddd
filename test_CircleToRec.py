import cv2
import numpy as np
from PIL import Image
import math


def get_huan_by_circle(img, circle_center, radius, radius_width):
    black_img = np.zeros((radius_width, int(2 * radius * math.pi), 3), dtype='uint8')
    for row in range(0, black_img.shape[0]):
        for col in range(0, black_img.shape[1]):
            theta = math.pi * 2 / black_img.shape[1] * (col + 1)
            rho = radius - black_img.shape[0] + row - 1
            p_x = int(circle_center[0] + rho * math.sin(math.pi * 2 - theta) + 0.5) - 1
            p_y = int(circle_center[1] - rho * math.cos(math.pi * 2 - theta) + 0.5) - 1
            black_img[row, col, :] = img[p_y, p_x, :]
    # cv2.imshow('bk', black_img)
    # cv2.waitKey()
    # cv2.imwrite('bk1.jpg', black_img)
    return black_img


img = cv2.imread('stamp_photo/0_stamp.jpg')
# get_huan_by_circle(img,(305,305),305,131,-1*math.pi/2)
out_img = get_huan_by_circle(img, (152, 152), 152, 100)
cv2.imshow('out_img',out_img)
cv2.waitKey(0)
