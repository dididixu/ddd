import os
import random
import shutil

rootdir = r"D:\dataset\dataset-desktop"
classes = ["clean", "messy"]

for label in classes:
    img_filepath = os.path.join(rootdir, label)

    temp_img = os.listdir(img_filepath)
    total_img = []
    for img in temp_img:
        if img.endswith(".jpg"):
            total_img.append(img)

    trainval_percent = 0.9
    train_percent = 0.8

    num = len(temp_img)
    list = range(num)

    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    if not os.path.exists(os.path.join(rootdir, "trainval", label)):
        os.makedirs(os.path.join(rootdir, "trainval", label))
    if not os.path.exists(os.path.join(rootdir, "train", label)):
        os.makedirs(os.path.join(rootdir, "train", label))
    if not os.path.exists(os.path.join(rootdir, "val", label)):
        os.makedirs(os.path.join(rootdir, "val", label))
    if not os.path.exists(os.path.join(rootdir, "test", label)):
        os.makedirs(os.path.join(rootdir, "test", label))

    for i in list:
        name = total_img[i]
        if i in trainval:
            shutil.copy(os.path.join(img_filepath, name), os.path.join(rootdir, "trainval", label))
            if i in train:
                shutil.copy(os.path.join(img_filepath, name), os.path.join(rootdir, "train", label))
            else:
                shutil.copy(os.path.join(img_filepath, name), os.path.join(rootdir, "val", label))
        else:
            shutil.copy(os.path.join(img_filepath, name), os.path.join(rootdir, "test", label))
