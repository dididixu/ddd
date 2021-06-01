#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
import logging
import time, json, re, os, shutil
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_dropzone import Dropzone

app = Flask(__name__)
dropzone = Dropzone(app)
# app.config['UPLOAD_FOLDER'] = 'interface/upload/uploads_photo'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg'}  # 集合类型
app.config["JSON_AS_ASCII"] = False
# app.config['UPLOADED_PATH'] = os.path.join(app.root_path, 'upload')
app.config['ALLOWED_EXTENSIONS_VIDEO'] = {'mp4'}  # 集合类型
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'interface', 'upload',
                                                  'uploads_photo')  # you'll need to create a folder named uploads
app.config['UPLOADED_AVI_DEST'] = os.path.join(basedir, 'interface', 'upload', 'uploads_avi')

photos = UploadSet('photos', IMAGES)
avis = UploadSet('avi', IMAGES)
configure_uploads(app, photos)
configure_uploads(app, avis)
patch_request_class(app)  # set maximum file size, default is 16MB


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def compare_img_p_hash(img1, img2):
    '''
    ham_dist == 0 -> particularly like
    ham_dist < 5  -> very like
    ham_dist > 10 -> different image
    '''

    hash_img1 = get_img_p_hash(img1)
    hash_img2 = get_img_p_hash(img2)

    similarity = ham_dist_func(hash_img1, hash_img2)

    print(similarity)

    return similarity


def get_img_p_hash(img):
    ## param img: img in MAT format (img = cv2.imread(image))

    hash_len = 32

    # GET Gray image
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Resize image, use a different way to obtain the best result
    resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_AREA)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_LANCZOS4)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_LINEAR)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_NEAREST)
    # resize_gray_img = cv.resize(gray_img, (hash_len, hash_len), cv.INTER_CUBIC)

    # Change int format of the image to float, for better DCT
    h, w = resize_gray_img.shape[:2]
    vis0 = np.zeros((h, w), np.float32)
    vis0[:h, :w] = resize_gray_img

    # DCT: Discrete cosine transform
    vis1 = cv.dct(cv.dct(vis0))
    vis1.resize(hash_len, hash_len)
    img_list = vis1.flatten()

    # Calculate the avg value
    avg = sum(img_list) * 1. / len(img_list)
    avg_list = []
    for i in img_list:
        if i < avg:
            tmp = '0'
        else:
            tmp = '1'
        avg_list.append(tmp)

    # Calculate the hash value
    p_hash_str = ''
    for x in range(0, hash_len * hash_len, 4):
        p_hash_str += '%x' % int(''.join(avg_list[x:x + 4]), 2)
    return p_hash_str


def ham_dist(x, y):
    assert len(x) == len(y)
    return sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])


def ham_dist_func(x, y):
    assert len(x) == len(y)
    ham_value = sum([ch1 != ch2 for ch1, ch2 in zip(x, y)])

    print(ham_value)

    ham_trans = 1.5 / (1.5 + ham_value)
    return ham_trans


logger = logging.getLogger(__name__)


def feature_similarity(img1, img2):
    img1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)

    similarity = orb_similarity(img1, img2)

    print(similarity)
    return similarity


def orb_similarity(img1, img2):
    """
    ORB algo similarity
    """
    try:
        # initialization
        orb = cv.ORB_create()
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        ##print(type(des1))

        # obtain the b.f.matcher
        bf = cv.BFMatcher(cv.NORM_HAMMING)

        # knn to select the results
        matches = bf.knnMatch(des1, trainDescriptors=des2, k=2)

        # examine the numbers of best matches
        good = [m for (m, n) in matches if m.distance < 0.78 * n.distance]
        similarity = len(good) / len(matches)
        return similarity

    except Exception as e:
        logger.info(e)
        return '0'


# files_list = []
# files_list_map = {}
#
# @app.route('/', methods=['GET', 'POST'])
# def manage_file_index():
#     files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
#     files_list_dir = []
#     files_str = str(files_list)
#     for i in files_list:
#         if ('_out_' not in i and len(i.split('_')[0]) == 1):
#             files_list_dir.append(i)
#
#     for i in files_list:
#         num = i.split('_')
#         if ('_out_' in i and len(num[0]) == 1 and len(num[1]) == 1):
#             files_list_map[re.sub('\d_out_', '', i)] = i
#     return render_template('manage_jpg.html', files_list=files_list_dir, files_map=files_list_map)


@app.route("/orb", methods=["POST"])
def orb():
    Res = {}
    try:
        img1_url = request.files['image1']
        img2_url = request.files['image2']
        if img1_url and allowed_file(img1_url.filename) and img2_url and allowed_file(img2_url.filename):
            filename1 = img1_url.filename
            savepath1 = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename1)
            filename2 = img2_url.filename
            savepath2 = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename2)
            img1_url.save(savepath1)
            img2_url.save(savepath2)
            img1 = cv.imread(savepath1)
            img2 = cv.imread(savepath2)
            result = feature_similarity(img1, img2)
            Res['Similarity'] = result
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        return json_res
    except:
        return "warning!"


@app.route("/hash", methods=["POST"])
def hash():
    Res = {}
    try:
        img1_url = request.files['image1']
        img2_url = request.files['image2']
        if img1_url and allowed_file(img1_url.filename) and img2_url and allowed_file(img2_url.filename):
            filename1 = img1_url.filename
            savepath1 = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename1)
            filename2 = img2_url.filename
            savepath2 = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename2)
            img1_url.save(savepath1)
            img2_url.save(savepath2)
            img1 = cv.imread(savepath1)
            img2 = cv.imread(savepath2)
            result = compare_img_p_hash(img1, img2)
            Res['Similarity'] = result
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        return json_res
    except:
        return "warning!"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10306, debug=True, processes=True)
    ##print(result)
