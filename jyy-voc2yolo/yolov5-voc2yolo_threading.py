import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import shutil
import threading
import time

sets = ['train', 'val']

# classes = ['drink', 'eat', 'hori', 'call', 'vert', 'smoke', 'stand', 'normal', 'other', 'lie', 'grovel']
# classes = ['shirt', 'non_shirt', 'western_style_clothes', 'coat', 'down_filled_coat', 'cotton', 'sweater', 'silk_scarf',
#            'tie', 'bow_tie']
# classes = ["eat"]
#
classes = ["coat",
           "western_style_clothes",
           "cotton",
           "non_shirt",
           "shirt",
           "sweater",
           "long_hair",
           "braid",
           "regular",
           "bald",
           ]

classes_dict = {"coat": 1,
           "western_style_clothes": 0,
           "cotton": 1,
           "non_shirt": 1,
           "shirt": 0,
           "sweater": 1,
           "long_hair": 3,
           "braid": 2,
           "regular": 2,
           "bald": 2,
           }


def get_img(image_set, out_photo_path):
    image_ids = open('ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    for image_id in image_ids:
        src_jpg = 'JPEGImages/%s.jpg' % (image_id)
        dst_jpg = out_photo_path + '/%s.jpg' % (image_id)
        if os.path.isfile(src_jpg) == 0:
            src_jpg = 'JPEGImages/%s.png' % (image_id)
            dst_jpg = out_photo_path + '/%s.png' % (image_id)
        if os.path.isfile(src_jpg) == 0:
            print("can not find " + image_id)
            continue
        imgs = []
        imgs.append(image_id)
        imgs.append(src_jpg)
        imgs.append(dst_jpg)
        try:
            yield imgs
        except:
            break


def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_set, image_id):
    in_file = open('Annotations/%s.xml' % (image_id))
    out_file = open('%s/labels/%s.txt' % (image_set, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue

        # change
        # cls_id = classes.index(cls)
        cls_id = classes_dict[cls]


        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


lock = threading.Lock()
photonum = threading.Semaphore(1)


def loop(image_set, imgs):
    print("thread %s is running!" % threading.current_thread().name)
    while True:
        try:
            with lock:
                image_id, src_jpg, dst_jpg = next(imgs)
        except StopIteration:
            break
        try:
            convert_annotation(image_set, image_id)
            # print(src_jpg, "\n", dst_jpg)
            shutil.copyfile(src_jpg, dst_jpg)
        except:
            print("change %s false", src_jpg)
    print("thread %s is end!" % threading.current_thread().name)


def main():
    thread_num = 20
    for image_set in sets:
        out_path = '%s/labels' % (image_set)
        out_photo_path = '%s/images' % (image_set)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        if not os.path.exists(out_photo_path):
            os.makedirs(out_photo_path)
        imgs = get_img(image_set, out_photo_path)
        for ii in range(thread_num):
            t = threading.Thread(target=loop, name='LoopThread %s' % ii, args=(image_set, imgs,))
            t.start()


main()
