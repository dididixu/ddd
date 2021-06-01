import cv2
import numpy as np
from utils import LinearTran, turncolorBG, enchcolor
import os

_DILATE_KERNEL = np.array([[0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 1, 0, 0],
                           [0, 0, 1, 0, 0]], dtype=np.uint8)

def enhance_contrast(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_output


def dilate(img):
    dilated = cv2.dilate(img, _DILATE_KERNEL)
    return dilated


def EFS(path):
    img = enchcolor(file_name=path)
    timg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, timg = cv2.threshold(timg, timg.mean(), 255, cv2.THRESH_BINARY)  # 做MASK与原图操作，背景变白，字体公章不变
    b = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    g = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)
    r = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    b[:, :] = img[:, :, 0]  # 复制 b 通道的数据
    g[:, :] = img[:, :, 1]  # 复制 g 通道的数据
    r[:, :] = img[:, :, 2]  # 复制 r 通道的数据

    tr = turncolorBG(r, timg)
    tb = turncolorBG(b, timg)
    tg = turncolorBG(g, timg)
    pimg = np.dstack((tb, tg, tr))
    # cv2.imshow('shen', pimg)
    # cv2.waitKey(0)
    # pimg = enhance_contrast(pimg)
    cv2.imshow('shen', pimg)
    cv2.waitKey(0)
    img = cv2.cvtColor(pimg, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array([156, 10, 10])
    upper_hsv = np.array([180, 255, 255])
    mask1 = cv2.inRange(img, lower_hsv, upper_hsv)
    lower_hsv = np.array([0, 10, 10])
    upper_hsv = np.array([20, 255, 255])
    mask2 = cv2.inRange(img, lower_hsv, upper_hsv)

    mask3 = mask1 + mask2
    mask3 = dilate(mask3)
    _, mask3 = cv2.threshold(mask3, mask3.mean(), 255, cv2.THRESH_BINARY)
    re1 = img[:, :, 0] * mask3
    # _, re1 = cv2.threshold(re1, re1.mean(), 255, cv2.THRESH_BINARY)

    re2 = img[:, :, 1] * mask3
    re3 = img[:, :, 2] * mask3
    re = np.dstack((re1, re2, re3))
    re = 255 - re
    re = cv2.cvtColor(re,cv2.COLOR_HSV2BGR)
    re = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)
    cv2.imshow('re', re)

    cv2.waitKey()
    cv2.destroyAllWindows()


EFS('11.jpg')
