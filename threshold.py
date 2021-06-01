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
    red_low_mask = cv2.inRange(hue_image, np.array([0, 20, 20]), np.array([15, 255, 255]))
    red_high_mask = cv2.inRange(hue_image, np.array([155, 20, 20]), np.array([180, 255, 255]))
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


def smoothing(image):
    # image = cv2.boxFilter(image, -1, (3, 3), normalize=1)
    image = cv2.blur(image, (3, 3))
    image = cv2.boxFilter(image, -1, (3, 3), normalize=1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = MedianFilter(image, 3)
    # image = MedianFilter(image, 3)
    # image = MedianFilter(image, 3)
    return image


def threshold(img):
    t2, Otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Otsu


def solve_stamp(img):
    img = enhance_contrast(img)
    img = mask(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = smoothing(img)
    # _, img = cv2.threshold(img, 127, 255,0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img


def fillHole(im_in):
    im_floodfill = im_in.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_in.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = im_in | im_floodfill_inv
    return im_out


if __name__ == '__main__':
    img = cv2.imread('5.png')
    img = solve_stamp(img)
    cv2.imshow('1234', img)
    cv2.imwrite('123.jpg', img)
    cv2.waitKey(0)
