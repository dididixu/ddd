# coding=utf-8
import cv2
import numpy as np
from pylab import *
np.set_printoptions(threshold=np.inf)
# Enhancing the contrast加强对比度
def enhance_contrast(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output

def mask(image):
    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red_low_mask = cv2.inRange(hue_image, np.array([0, 10, 10]), np.array([60, 255, 255]))
    red_high_mask = cv2.inRange(hue_image, np.array([155, 10, 10]), np.array([180, 255, 255]))
    index1 = red_low_mask == 255
    index2 = red_high_mask == 255
    index3 = index1 | index2
    img_sum = np.zeros(image.shape, np.uint8)
    img_sum[:, :] = (255, 255, 255)
    img_sum[index3] = image[index3]  # (0,0,255)
    return img_sum

def MedianFilter(Image, dim):  # Image为待处理图像，dim为滤波器的大小dim*dim
    im = array(Image)
    sigema = []
    for i in range(int(dim / 2), im.shape[0] - int(dim / 2)):
        for j in range(int(dim / 2), im.shape[1] - int(dim / 2)):
            for a in range(-int(dim / 2), -int(dim / 2) + dim):
                # for b in range(-int(dim / 2), -int(dim / 2) + dim):
                if a != 0:
                    sigema.append(Image[i, j + a])
                    sigema.append(Image[i + a, j])
                else:
                    sigema.append(Image[i, j])
            sigema.sort()
            im[i, j] = sigema[int((2 * dim - 1) / 2)]
            sigema = []
    return im

cv2.imwrite()
def smoothing(img):
    image1 = cv2.boxFilter(img, -1, (3, 3), normalize=1)
    image2 = cv2.boxFilter(image1, -1, (3, 3), normalize=1)
    image3 = MedianFilter(image2, 3)
    return image3

def solve(img):
    img = enhance_contrast(img)
    img = mask(img)
    img = smoothing(img)
    return img