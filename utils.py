import pandas as pd
import numpy as np
from PIL import Image,ImageEnhance
from skimage import io,img_as_ubyte
import cv2

def enchcolor(file_name):
    img = io.imread(file_name)

    img = img * 1.0
    img_out = img * 1.0

    # -1 ~ 1

    Increment = 0.5

    img_min = img.min(axis=2)
    img_max = img.max(axis=2)

    Delta = (img_max - img_min) / 255.0
    value = (img_max + img_min) / 255.0
    L = value / 2.0

    mask_1 = L < 0.5

    s1 = Delta / (value + 0.001)
    s2 = Delta / (2 - value + 0.001)
    s = s1 * mask_1 + s2 * (1 - mask_1)

    if Increment >= 0:
        temp = Increment + s
        mask_2 = temp > 1
        alpha_1 = s
        alpha_2 = s * 0 + 1 - Increment
        alpha = alpha_1 * mask_2 + alpha_2 * (1 - mask_2)
        alpha = 1 / (alpha + 0.001) - 1
        img_out[:, :, 0] = img[:, :, 0] + (img[:, :, 0] - L * 255.0) * alpha
        img_out[:, :, 1] = img[:, :, 1] + (img[:, :, 1] - L * 255.0) * alpha
        img_out[:, :, 2] = img[:, :, 2] + (img[:, :, 2] - L * 255.0) * alpha

    else:
        alpha = Increment
        img_out[:, :, 0] = L * 255.0 + (img[:, :, 0] - L * 255.0) * (1 + alpha)
        img_out[:, :, 1] = L * 255.0 + (img[:, :, 1] - L * 255.0) * (1 + alpha)
        img_out[:, :, 2] = L * 255.0 + (img[:, :, 2] - L * 255.0) * (1 + alpha)

    img_out = img_out / 255.0

    # 饱和处理
    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2

    new = img_as_ubyte(img_out)
    new = cv2.cvtColor(new,cv2.COLOR_RGB2BGR)
    return new


def enrenge():
    path = 'teres.jpg'
    img = Image.open(path)

    enhancer3 = ImageEnhance.Contrast(img)  # 用 Contrast 增强对比度
    enhancer_img3 = enhancer3.enhance(0.5)
    enhancer_img3.save("test1.jpg")


def ExpTran(im, esp=0, gama=1):
    imarray = np.array(im)
    height, width = imarray.shape

    for i in range(height):
        for j in range(width):
            tmp = imarray[i, j] / 255
            tmp = int(pow(tmp + esp, gama) * 255)
            if tmp >= 0 and tmp <= 255:
                imarray[i, j] = tmp
            elif tmp > 255:
                imarray[i, j] = 255
            else:
                imarray[i, j] = 0
    return imarray

def ColorsSave(colours, name):
    data = pd.DataFrame(colours)
    writer = pd.ExcelWriter(name + ".xlsx")
    data.to_excel(writer, "page_1", float_format='%d')
    writer.save()

    writer.close()

def LinearTran(im, a=1, b=0):
    imarray = np.array(im)

    try:
        height, width = imarray.shape
        for i in range(height):
            for j in range(width):
                aft = int(a * imarray[i, j] + b)
                if aft <= 255 and aft >= 0:
                    imarray[i, j] = aft
                elif aft > 255:
                    imarray[i, j] = 255
                else:
                    imarray[i, j] = 0
        return imarray
    except:
        print('imarray have no  attribute name shape')


def median_filter(img, K_size=3):
    H, W, C = img.shape

    ## Zero padding

    pad = K_size // 2

    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)

    out[pad:pad + H, pad:pad + W] = img.copy().astype(np.float)

    tmp = out.copy()

    # filtering

    for y in range(H):

        for x in range(W):

            for c in range(C):
                out[pad + y, pad + x, c] = np.median(tmp[y:y + K_size, x:x + K_size, c])

    out = out[pad:pad + H, pad:pad + W].astype(np.uint8)

    return out


def getVarianceMean(scr, winSize):
    if scr is None or winSize is None:
        print("The input parameters of getVarianceMean Function error")
        return -1

    if winSize % 2 == 0:
        print("The window size should be singular")
        return -1

    copyBorder_map = cv2.copyMakeBorder(scr, winSize // 2, winSize // 2, winSize // 2, winSize // 2,
                                        cv2.BORDER_REPLICATE)
    shape = np.shape(scr)

    local_mean = np.zeros_like(scr)
    local_std = np.zeros_like(scr)

    for i in range(shape[0]):
        for j in range(shape[1]):
            temp = copyBorder_map[i:i + winSize, j:j + winSize]
            local_mean[i, j], local_std[i, j] = cv2.meanStdDev(temp)
            if local_std[i, j] <= 0:
                local_std[i, j] = 1e-8

    return local_mean, local_std

def adaptContrastEnhancement(scr, winSize, maxCg):
    if scr is None or winSize is None or maxCg is None:
        print("The input parameters of ACE Function error")
        return -1

    YUV_img = cv2.cvtColor(scr, cv2.COLOR_BGR2YUV)  ##转换通道
    Y_Channel = YUV_img[:, :, 0]
    shape = np.shape(Y_Channel)

    meansGlobal = cv2.mean(Y_Channel)[0]

    ##这里提供使用boxfilter 计算局部均质和方差的方法
    #    localMean_map=cv2.boxFilter(Y_Channel,-1,(winSize,winSize),normalize=True)
    #    localVar_map=cv2.boxFilter(np.multiply(Y_Channel,Y_Channel),-1,(winSize,winSize),normalize=True)-np.multiply(localMean_map,localMean_map)
    #    greater_Zero=localVar_map>0
    #    localVar_map=localVar_map*greater_Zero+1e-8
    #    localStd_map = np.sqrt(localVar_map)

    localMean_map, localStd_map = getVarianceMean(Y_Channel, winSize)

    for i in range(shape[0]):
        for j in range(shape[1]):

            cg = 0.2 * meansGlobal / max(localStd_map[i, j],1)
            if cg > maxCg:
                cg = maxCg
            elif cg < 1:
                cg = 1

            temp = Y_Channel[i, j].astype(float)
            temp = max(0, min(localMean_map[i, j] + cg * (temp - localMean_map[i, j]), 255))

            #            Y_Channel[i,j]=max(0,min(localMean_map[i,j]+cg*(Y_Channel[i,j]-localMean_map[i,j]),255))
            Y_Channel[i, j] = temp

    YUV_img[:, :, 0] = Y_Channel

    dst = cv2.cvtColor(YUV_img, cv2.COLOR_YUV2BGR)

    return dst


def duibi(img):
    if img is None:
        print("The file name error,please check it")
        return -1

    print(np.shape(img))
    dstimg = adaptContrastEnhancement(img, 15, 10)

    cv2.imshow('output.jpg', dstimg)

    out1 = median_filter(img, K_size=3)
    cv2.imshow("out1.jpg", out1)

    cv2.waitKey(0)

    return 0


def quchubei(imgp):
    farina = cv2.imread(imgp, 0)

    Imax = np.max(farina)
    Imin = np.min(farina)
    MAX = 255
    MIN = 0
    farina_cs = (farina - Imin) / (Imax - Imin) * (MAX - MIN) + MIN
    cv2.imshow("farina_cs", farina_cs.astype("uint8"))
    cv2.waitKey()

def addpic(img1,img2):
    img = np.zeros((img1.shape[0], img1.shape[1]), dtype=img1.dtype)

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if (int(img1[i][j]) >int(img2[i][j])):
                img[i][j] = img2[i][j]
            else:
                img[i][j] = img1[i][j]
    return img

def turncolorBG(img,mask):
    Timg = np.zeros((img.shape[0], img.shape[1]), dtype=img.dtype)

    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(mask[i][j] == 0):
                Timg[i][j] = img[i][j]
            else:
                Timg[i][j] = mask[i][j]
    return Timg


