# coding=utf-8
import cv2
import matplotlib.pyplot as plt
from pylab import *


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



def MedianFilter(Imge,dim):       #Image为待处理图像，dim为滤波器的大小dim*dim
    im=array(Imge)
    sigema=[]
    for i in range(int(dim/2), im.shape[0] - int(dim/2)):
        for j in range(int(dim/2), im.shape[1] - int(dim/2)):
            for a in range(-int(dim/2), -int(dim/2)+dim):
                for b in range(-int(dim/2), -int(dim/2)+dim):
                    sigema.append(img[i + a, j + b])
            sigema.sort()
            im[i, j] = sigema[int(dim*dim/2)]
            sigema = []
    return im

def solve(imagepath):
    img = cv2.imread(imagepath, 0)
    # height, width, channel = image.shape

    image1 = cv2.boxFilter(img, -1, (3, 3), normalize=1)
    image2 = cv2.boxFilter(image1, -1, (3, 3), normalize=1)
    image3 = MedianFilter(image2, 3)
    # image = cv2.boxFilter(image, -1, (
    # 3, 3))
    return img, image1, image2, image3
    # plt.subplot(2, 2, 1), plt.imshow(img, 'gray')  # 默认彩色，另一种彩色bgr
    # plt.subplot(2, 2, 2), plt.imshow(image1, 'gray')
    # plt.subplot(2, 2, 3), plt.imshow(image2, 'gray')
    # plt.subplot(2, 2, 4), plt.imshow(image3, 'gray')
    # plt.show()
    #
    # plt.figure(figsize=(20, 10))
    # plt.imshow(image)
    # plt.show()

# cv2.threshold(cv2.THRESH_OTSU)


if __name__ == '__main__':
    imagepath = '3.jpg'
    img = cv2.imread(imagepath, 0)
    solve(img)
