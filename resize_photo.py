# coding=utf-8
from PIL import Image
import os

# import Image
size = 320, 320


def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)


if __name__ == '__main__':
    photo_path = r"C:\Users\49429\Desktop\zhaopian"
    save_path = r"C:\Users\49429\Desktop\123"
    imglist = os.listdir(photo_path)
    for img in imglist:
        # print(img)
        im = Image.open(os.path.join(photo_path, img))
        im.thumbnail(size, Image.ANTIALIAS)
        im.save(os.path.join(save_path, img), "png")
