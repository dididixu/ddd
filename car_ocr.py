# ## -*- coding: utf-8 -*-
import cv2
import imutils
import numpy as np


def RGB_GRAY(img):
    img = cv2.resize(img, (640, 480))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img2', gray)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    return img, gray


def edge(img):
    #  cv2.Canny（source_image，thresholdValue 1，thresholdValue 2）
    edged = cv2.Canny(img, 30, 200)
    cv2.imshow('img', edged)
    cv2.imshow('img1', img)
    cv2.waitKey(0)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    return contours


def draw(img, gray, cnt):
    screenCnt = None
    for c in cnt:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break
    if screenCnt is None:
        detected = 0
        # print("No contour detected")
        return img, None
    else:
        detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('new_image', new_image)
    # cv2.waitKey(0)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
    return img, Cropped


def show(img, Cropped):
    # text = pytesseract.image_to_string(Cropped, config='--psm 11')
    # print("programming_fever's License Plate Recognition\n")
    # print("Detected license plate Number is:", text)
    img = cv2.resize(img, (500, 300))
    Cropped = cv2.resize(Cropped, (400, 200))
    cv2.imshow('car', img)
    cv2.imshow('Cropped', Cropped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def solve(img):
    img, gray = RGB_GRAY(img)
    contours = edge(gray)
    img, Cropped = draw(img, gray, contours)
    if Cropped == None:
        print("No contour detected!")
        return
    show(img, Cropped)


if __name__ == '__main__':
    img_path = './33333.jpg'
    img = cv2.imread(img_path)
    solve(img)
