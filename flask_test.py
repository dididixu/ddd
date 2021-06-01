# -*- coding: utf-8 -*-
import hashlib
import os
from flask import Flask, request
import json
import time
import cv2
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
app.config['ALLOWED_EXTENSIONS_VIDEO'] = {'mp4', 'avi'}  # 集合类型
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


def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS_VIDEO']


@app.route("/pred_img", methods=["POST"])
def pred_ocr():
    Res = {}

    upload_file = request.files['image']
    print(upload_file)
    # res = text_predict(img_path, model_type='cnocr')
    if upload_file and allowed_file(upload_file.filename):
        filename = upload_file.filename
        savepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        print(savepath)
        upload_file.save(savepath)
        s_time = time.time()
        # res = predict(savepath)
        res = 1
        cost_time = time.time() - s_time
        Res['colour'] = res
        # Res['cost_time'] = cost_time
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        print("res:", json_res)
        os.remove(savepath)
        return json_res
    else:
        return None


@app.route('/ddd', methods=["POST"])
def ddd():
    # upload_file = request.files['image']
    if request.method == 'POST':
        print(len(request.files.getlist('image')))
        for f in request.files.getlist('image'):
            f.save(os.path.join(app.config['UPLOADED_PHOTOS_DEST'], f.filename))
            print(f.filename)
    # return render_template('index.html')
    return "hello"


@app.route("/pred_video", methods=["POST"])
def pred_video():
    Res = {}

    upload_file = request.files['image']
    print(upload_file)
    # res = text_predict(img_path, model_type='cnocr')
    if upload_file and allowed_file_video(upload_file.filename) and upload_file.filename:
        filename = upload_file.filename
        savepath = os.path.join(app.config['UPLOADED_AVI_DEST'], filename)
        print(savepath)
        upload_file.save(savepath)
        s_time = time.time()
        # res = predict(savepath)
        cap = cv2.VideoCapture(savepath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        outpath = os.path.join(app.config['UPLOADED_AVI_DEST'],
                               'out_' + str(filename.strip().split(".")[0:-1])[2:-2] + '.avi')
        out = cv2.VideoWriter(outpath, fourcc, 20.0, (640, 480))
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            # cv2.imshow('frame', frame)
            # time.sleep(0.04)
            out.write(frame)
    return "ok"


@app.route('/clear_output', methods=["post"])
def clear_output():
    import shutil

    video_filepath = app.config['UPLOADED_AVI_DEST']
    photo_filepath = app.config['UPLOADED_PHOTOS_DEST']
    try:
        shutil.rmtree(video_filepath)
        os.mkdir(video_filepath)
        shutil.rmtree(photo_filepath)
        os.mkdir(photo_filepath)
        return "clear successful!"
    except:
        return "clear unsuccessful!"


@app.route('/pred_statistics', methods=["post"])
def pred_statistics():
    result = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    from sklearn.metrics import confusion_matrix, classification_report
    y_true = []
    y_pred = []
    target_names = ['黑屏', '白色', '灰色']
    for file in result:
        label, predict = file.split("_")[0], file.split("_")[1]
        if (len(label) == 1 and len(predict) == 1):
            y_true.append(label)
            y_pred.append(predict)
    # s = confusion_matrix(y_true, y_pred)
    return classification_report(y_true, y_pred, target_names=target_names)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    form = UploadForm()
    if form.validate_on_submit():
        for filename in request.files.getlist('photo'):
            c = ('admin' + str(time.time())).encode("utf-8")
            name = hashlib.md5(c).hexdigest()[:15]
            photos.save(filename, name=name + '.')
        success = True
    else:
        success = False
    return render_template('index1.html', form=form, success=success)


files_list = []
files_list_map = {}
import re


@app.route('/manage_jpg')
def manage_file():
    files_list_map = {}
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    files_list_dir = []

    files_str = str(files_list)
    for i in files_list:
        if ('_out_' not in i and len(i.split('_')[0]) == 1):
            files_list_dir.append(i)

    for i in files_list:
        num = i.split('_')
        if ('_out_' in i and len(num[0]) == 1 and len(num[1]) == 1):
            files_list_map[re.sub('\d_out_', '', i)] = i
    return render_template('manage_jpg.html', files_list=files_list_dir, files_map=files_list_map)


@app.route('/manage_avi')
def manage_file1():
    files_list = os.listdir(app.config['UPLOADED_AVI_DEST'])
    return render_template('manage_avi.html', files_list=files_list)


@app.route('/open/<filename>')
def open_file(filename):
    file_url = photos.url(filename)
    return render_template('browser.html', file_url=file_url)


@app.route('/open1/<filename>')
def open_file1(filename):
    file_url = avis.url(filename)
    return render_template('browser.html', file_url=file_url)


@app.route('/open2/<filename>')
def open_file2(filename):
    if (files_list_map.get(filename, 0) != 0):
        file_url = photos.url(files_list_map[filename])
        return render_template('browser.html', file_url=file_url)
    else:
        return render_template('browser1.html', file_url="", state='no exist photo')


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_file'))


@app.route('/delete1/<filename>')
def delete_file1(filename):
    file_path = avis.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_file1'))


@app.route('/delete2/<filename>')
def delete_file2(filename):
    if (files_list_map.get(filename, 0) != 0):
        file_path = photos.path(files_list_map[filename])
        files_list_map[filename] = ''
        os.remove(file_path)
        return redirect(url_for('manage_file'))
    else:
        return redirect(url_for('manage_file'))


@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'])
    return send_from_directory(file_path, filename, as_attachment=True)


@app.route('/download1/<filename>')
def download_file1(filename):
    file_path = os.path.join(app.config['UPLOADED_AVI_DEST'])
    return send_from_directory(file_path, filename, as_attachment=True)


@app.route('/download2/<filename>')
def download_file2(filename):
    if (files_list_map.get(filename, 0) != 0):
        file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'])
        return send_from_directory(file_path, files_list_map[filename], as_attachment=True)
    else:
        return render_template('browser1.html', file_url="", state='no exist photo')


if __name__ == '__main__':
    app.run(debug=True, port=8008)
