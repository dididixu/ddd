# coding: utf-8
import numpy as np
import random
import cv2
import glob
import xml.etree.cElementTree as ET
import scipy.misc as misc
from xml.etree.ElementTree import ElementTree, Element


# 随机平移
def random_translate(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (w_img, h_img))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
    return img, bboxes


# 随机裁剪
def random_crop(img, bboxes, p=0.5):
    if random.random() < p:
        h_img, w_img, _ = img.shape
        # 得到可以包含所有bbox的最大bbox
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w_img - max_bbox[2]
        max_d_trans = h_img - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        img = img[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
    return img, bboxes


# 随机水平反转
def random_horizontal_flip(img, bboxes=None, p=0.5):
    if random.random() < p:
        _, w_img, _ = img.shape
        # print(img.shape)
        # img = img[:, ::-1, :]
        imgs = cv2.flip(img, 1, dst=None)  # 水平镜像
        if bboxes is None:
            return imgs
        bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return imgs, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机垂直反转
def random_vertical_flip(img, bboxes, p=0.5):
    from time import sleep

    h_img, _, _ = img.shape
    img = img[::-1, :, :]

    bboxes[:, [1, 3]] = h_img - bboxes[:, [3, 1]]

    return img, bboxes


def letterbox(img, new_shape=(640, 640), bboxes=None, color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if bboxes is None:
        return img, ratio, (dw, dh)
    else:
        bboxes[:, [0]] = bboxes[:, [0]] * r + dw
        bboxes[:, [1]] = bboxes[:, [1]] * r + dh
        bboxes[:, [2]] = bboxes[:, [2]] * r + dw
        bboxes[:, [3]] = bboxes[:, [3]] * r + dh
        return img, bboxes, ratio, (dw, dh)


# 随机顺时针旋转90
def random_rot90_1(img, bboxes=None, p=0.5):
    '''
    :param img: nparray img
    :param bboxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
    :param p: 随机比例
    :return:
    '''
    # 顺时针旋转90度
    if random.random() < p:
        h, w, _ = img.shape
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 1)
        if bboxes is None:
            return new_img
        else:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            bboxes[:, [0, 2]] = h - bboxes[:, [0, 2]]
            return new_img, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机逆时针旋转
def random_rot90_2(img, bboxes=None, p=0.5):
    '''
    :param img: nparray img
    :param bboxes: np.array([[88, 176, 250, 312, 1222], [454, 115, 500, 291, 1222]]), 里面为x1, y1, x2, y2, 标签
    :param p: 随机比例
    :return:
    '''
    # 逆时针旋转90度
    if random.random() < p:
        h, w, _ = img.shape
        trans_img = cv2.transpose(img)
        new_img = cv2.flip(trans_img, 0)
        if bboxes is None:
            return new_img
        else:
            # bounding box 的变换: 一个图像的宽高是W,H, 如果顺时90度转换，那么原来的原点(0, 0)到了 (H, 0) 这个最右边的顶点了，
            # 设图像中任何一个转换前的点(x1, y1), 转换后，x1, y1是表示到 (H, 0)这个点的距离，所以我们只要转换回到(0, 0) 这个点的距离即可！
            # 所以+90度转换后的点为 (H-y1, x1), -90度转换后的点为(y1, W-x1)
            bboxes[:, [0, 1, 2, 3]] = bboxes[:, [1, 0, 3, 2]]
            bboxes[:, [1, 3]] = w - bboxes[:, [1, 3]]
            return new_img, bboxes
    else:
        if bboxes is None:
            return img
        else:
            return img, bboxes


# 随机对比度和亮度 (概率：0.5)
def random_bright(img, bboxes=None, p=0.5, lower=0.7, upper=0.8):
    if random.random() < p:
        mean = np.mean(img)
        img = img - mean
        img = img * random.uniform(lower, upper) + mean * random.uniform(lower, upper)  # 亮度
        # img = img / 255.
    if bboxes is None:
        return img
    return img, bboxes


# 随机变换通道
def random_swap(im, bboxes, p=0.5):
    perms = ((0, 1, 2), (0, 2, 1),
             (1, 0, 2), (1, 2, 0),
             (2, 0, 1), (2, 1, 0))
    if random.random() < p:
        swap = perms[random.randrange(0, len(perms))]
        # print(swap)

        im[:, :, (0, 1, 2)] = im[:, :, swap]
    return im, bboxes


# 随机变换饱和度
def random_saturation(im, bboxes, p=0.5, lower=0.5, upper=1.5):
    if random.random() < p:
        im[:, :, 1] = im[:, :, 1] * random.uniform(lower, upper)
    return im, bboxes


# 随机变换色度(HSV空间下(-180, 180))
def random_hue(im, bboxes, p=0.5, delta=18.0):
    if random.random() < p:
        im[:, :, 0] = im[:, :, 0] + random.uniform(-delta, delta)
        im[:, :, 0][im[:, :, 0] > 360.0] = im[:, :, 0][im[:, :, 0] > 360.0] - 360.0
        im[:, :, 0][im[:, :, 0] < 0.0] = im[:, :, 0][im[:, :, 0] < 0.0] + 360.0
    return im, bboxes


# 随机旋转0-90角度
def random_rotate_image_func(image):
    # 旋转角度范围
    angle = np.random.uniform(low=0, high=90)
    return misc.imrotate(image, angle, 'bicubic')


def random_rot(img, boxes, angle, center=None, scale=1.0):
    import math

    height, width = img.shape[:2]

    # 绕着图片中心旋转得到的旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2), angle, 1)
    corner_points = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

    # 旋转后的角点
    lanmark_ = np.asarray([(rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2],
                            rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2]) for (x, y) in
                           corner_points])

    # 给图片填充边界
    right, bottom = np.max(lanmark_, axis=0)
    left, top = np.min(lanmark_, axis=0)

    top_pad = np.max((0, math.ceil(-top)))
    left_pad = np.max((0, math.ceil(-left)))
    bottom_pad = np.max((0, math.ceil(bottom - height)))
    right_pad = np.max((0, math.ceil(right - width)))

    img = cv2.copyMakeBorder(
        img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, 0)

    # 旋转图片
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2 + left_pad, height / 2 + top_pad), angle, 1)

    # 计算boxes

    boxes_container = []
    for index, box in enumerate(boxes):
        points = np.asarray([(box[0], box[1]), (box[0], box[3]), (box[2], box[1]), (box[2], box[3])])
        points = points + np.array([left_pad, top_pad])

        points = np.asarray([(rotation_matrix[0][0] * x + rotation_matrix[0][1] * y + rotation_matrix[0][2],
                              rotation_matrix[1][0] * x + rotation_matrix[1][1] * y + rotation_matrix[1][2]) for (x, y)
                             in points])
        tl_point = np.min(points, axis=0)
        rb_point = np.max(points, axis=0)
        boxes[index][0:4] = [tl_point[0], tl_point[1], rb_point[0], rb_point[1]]

    img_rotation = cv2.warpAffine(
        img, rotation_matrix, (img.shape[1], img.shape[0]))
    img = img_rotation
    return img, boxes


# 拷贝文件
def copy(src, dst, boxes, image_shape):
    # 同一地址无需拷贝
    if os.path.abspath(src) == os.path.abspath(dst):
        print('地址相同，无需拷贝')
        return
    # 判断是否是目录
    if os.path.isdir(src):
        print('源文件是目录文件，无法拷贝')
        return
    # 判断目标地址是否是目录
    if os.path.isdir(dst):
        # 提取源文件名
        src_name = os.path.basename(src)
        # 拼接目标文件名
        dst = os.path.join(dst, src_name)

    # 打开源文件
    src_fp = open(src, 'rb')
    # 打开目标文件
    dst_fp = open(dst, 'wb')
    while True:
        # 读取指定长度的内容
        content = src_fp.read(1024)
        # 判断是否读完
        if len(content) == 0:
            break
        # 将内容写入目标文件
        dst_fp.write(content)
    # 关闭源文件
    src_fp.close()
    # 关闭目标文件
    dst_fp.close()
    altAnnotations(dst, boxes, image_shape=image_shape)


def readAnnotations(xml_path):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    results = []
    for element_obj in element_objs:
        result = []
        class_name = element_obj.find('name').text

        obj_bbox = element_obj.find('bndbox')
        x1 = int(round(float(obj_bbox.find('xmin').text)))
        y1 = int(round(float(obj_bbox.find('ymin').text)))
        x2 = int(round(float(obj_bbox.find('xmax').text)))
        y2 = int(round(float(obj_bbox.find('ymax').text)))

        result.append(int(x1))
        result.append(int(y1))
        result.append(int(x2))
        result.append(int(y2))

        results.append(result)
    return results


def altAnnotations(xml_path, boxes, image_shape):
    et = ET.parse(xml_path)
    element = et.getroot()
    element_objs = element.findall('object')

    element.find('filename').text = xml_path.split('/')[-1][:-4] + '.jpg'
    try:
        element.find('path').text = xml_path[:-4] + '.jpg'
    except:
        pass

    for i, element_obj in enumerate(element_objs):
        obj_bbox = element_obj.find('bndbox')
        obj_bbox.find('xmin').text = str(min(boxes[i][0], boxes[i][2]))
        obj_bbox.find('ymin').text = str(min(boxes[i][1], boxes[i][3]))
        obj_bbox.find('xmax').text = str(max(boxes[i][0], boxes[i][2]))
        obj_bbox.find('ymax').text = str(max(boxes[i][1], boxes[i][3]))

    element_size = element.find('size')
    element_size.find('height').text = str(image_shape[0])
    element_size.find('width').text = str(image_shape[1])
    element_size.find('depth').text = str(image_shape[2])
    et.write(xml_path)


import os

if __name__ == "__main__":
    photo_type = ".jpg"  # 可以是.jpg或者.png
    img_path = r"D:\dataset\dataset-clothes\1229-final\JPEGImages"
    xml_path = r"D:\dataset\dataset-clothes\1229-final\Annotations"
    img_list = glob.glob(img_path + r"/*" + photo_type)  # 源图像路径
    img_save = r"D:\dataset\dataset-clothes\1229-final\enhancement\JPEGImages"  # 存储路径
    xml_save = r"D:\dataset\dataset-clothes\1229-final\enhancement\Annotations"

    if os.path.isdir(img_save) == 0:
        os.mkdir(img_save)
    if os.path.isdir(xml_save) == 0:
        os.mkdir(xml_save)
    i = len(os.listdir(img_save))
    for image_path in img_list:
        img_org = cv2.imread(image_path)
        img = img_org
        xml_path_save = os.path.join(xml_path, image_path.split('\\')[-1][:-4] + '.xml')
        print(xml_path_save)
        bboxes = readAnnotations(xml_path_save)
        print("img: {}".format(image_path))
        bboxes = np.array(bboxes)
        if len(bboxes) == 0:
            continue
        # 数据增强操作

        # img = letterbox(img, new_shape=(640, 640), bboxes=None)[0]
        # img = random_horizontal_flip(img, bboxes=None, p=1.0)  ###随机水平反转，最后参数为随机操作比例
        img, bboxes = random_horizontal_flip(img, bboxes=bboxes, p=0.5)  # 随机水平反转，最后参数为随机操作比例
        # # img, bboxes = random_vertical_flip(img, bboxes=bboxes, p=0.5)  # 随机垂直反转
        img = random_bright(img, p=0.5, lower=0.8, upper=1.1)
        img, bboxes = random_crop(img, bboxes=bboxes, p=0.6)  # 随机裁切
        img, bboxes = random_rot(img, boxes=bboxes, angle=np.random.randint(0, 90, 1)[0])  ###任意旋转角度
        img, bboxes = letterbox(img, new_shape=(640, 640), bboxes=bboxes)[0:2]  # resize 一定要放最后
        # img, bboxes = letterbox(img, new_shape=(480, 480), bboxes=np.array(bboxes))[0:2]
        # img, bboxes = random_rot90_1(img, bboxes, 0.5)  ###随机顺时针旋转90°
        # img = random_rot90_1(img, bboxes=None, p=0.5)  ###随机顺时针旋转90°
        # img, bboxes = random_rot90_1(img, np.array(bboxes), 1)            ###随机逆时针旋转90°
        # img, bboxes = random_translate(img, np.array(bboxes), 1)          ###随机平移 ****最好别用****

        # img = random_bright(img, p=0.5, lower=0.7, upper=1.3)  ###随机对比度和亮度
        # img, bboxes = random_swap(img, np.array(bboxes), 1)               ###随即通道转换
        # img, bboxes = random_saturation(img, np.array(bboxes), 1)         ###随即变换饱和度
        # img, bboxes = random_hue(img, np.array(bboxes), 1)                ###随机变换色度


        # 图像存储操作；xml修改存储操作
        # name = image_path.split("\\")[-1].split("_")[0]
        newpt = os.path.join(img_save, '{:0>8d}'.format(i))
        new_xml = os.path.join(xml_save, '{:0>8d}'.format(i))
        img = np.array(img)
        cv2.imwrite(newpt + photo_type, img)
        copy(src=xml_path_save, dst=new_xml + '.xml', boxes=bboxes, image_shape=img.shape)
        i += 1
        # 演示代码
        # for box in bboxes:
        #     cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        #
        # cv2.imshow(image_path, img)
        # img_rotate = 0
        # cv2.waitKey(0)
