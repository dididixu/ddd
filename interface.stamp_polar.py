## deployment code for python-flask interface deployment
import os
import time, math
import numpy as np
import sys, os, torch, mmcv
from mmcv.runner import load_checkpoint
from mmdet.apis import init_detector, inference_detector, show_result
import json

import time, json, re, cv2, os, shutil
from flask import Flask, render_template, redirect, Response, url_for, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_dropzone import Dropzone

##
from read_xml import GetAnnotBoxLoc

config_file = '/home/project/mmdetection_model_trained/stamp_result_0720/retinanet_r50_fpn_2x_coco.py'
checkpoint_file = '/home/project/mmdetection_model_trained/stamp_result_0720/epoch_120.pth'
model = init_detector(config_file, checkpoint_file)
app = Flask(__name__)
dropzone = Dropzone(app)
# app.config['UPLOAD_FOLDER'] = 'interface/upload/uploads_photo'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg'}  # 集合类型
app.config['ALLOWED_EXTENSIONS_XML'] = {'xml'}  # 集合类型
app.config['ALLOWED_EXTENSIONS_VIDEO'] = {'mp4'}  # 集合类型
app.config["JSON_AS_ASCII"] = False
# app.config['UPLOADED_PATH'] = os.path.join(app.root_path, 'upload')
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SECRET_KEY'] = 'I have a dream'
app.config['UPLOADED_PHOTOS_DEST'] = os.path.join(basedir, 'VOCdevkit', 'VOC2007',
                                                  'JPEGImages')  # you'll need to create a folder named uploads
app.config['UPLOADED_OUTPUT_DEST'] = os.path.join(basedir, 'VOCdevkit', 'VOC2007',
                                                  'output')  # you'll need to create a folder named uploads
app.config['UPLOADED_AVI_DEST'] = os.path.join(basedir, 'VOCdevkit', 'VOC2007',
                                               'avi')
app.config['UPLOADED_JSON_DEST'] = os.path.join(basedir, 'VOCdevkit', 'VOC2007',
                                                'json')
app.config['UPLOADED_XML_DEST'] = os.path.join(basedir, 'VOCdevkit', 'VOC2007',
                                               'Annotations')
app.config['UPLOADED_TXT_DEST'] = os.path.join(basedir, 'VOCdevkit', 'VOC2007',
                                               'ImageSets', 'Main')

photos = UploadSet('photos', IMAGES)
avis = UploadSet('avi', IMAGES)
jsons = UploadSet('json', IMAGES)
outputs = UploadSet('output', IMAGES)
xmls = UploadSet('xml', IMAGES)
txts = UploadSet('txt', IMAGES)
configure_uploads(app, photos)
configure_uploads(app, avis)
configure_uploads(app, jsons)
configure_uploads(app, outputs)
configure_uploads(app, xmls)
configure_uploads(app, txts)
patch_request_class(app)  # set maximum file size, default is 16MB


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class UploadForm(FlaskForm):
    photo = FileField(validators=[FileAllowed(photos, 'Image Only!'), FileRequired('Choose a file!')])
    submit = SubmitField('Upload')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


def allowed_xml(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS_XML']


def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS_VIDEO']


#############################################################
def predict(model, img, target_size):
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]


##########################################################
files_list = []
files_list_map = {}


@app.route('/', methods=['GET', 'POST'])
def manage_file_index():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    files_list_output = os.listdir(app.config['UPLOADED_OUTPUT_DEST'])
    files_list_dir = []
    for i in files_list:
        if ('_out_' not in i and len(i.split('_')[0]) == 1):
            files_list_dir.append(i)

    for i in files_list_output:
        num = i.split('_')
        if ('_out_' in i and len(num[0]) == 1 and len(num[1]) == 1):
            files_list_map[re.sub('\d_out_', '', i)] = i
    return render_template('manage_jpg.html', files_list=files_list_dir, files_map=files_list_map)


def get_huan_by_circle(img, circle_center, radius, radius_width):
    print(circle_center, radius)
    black_img = np.zeros((radius_width, int(2 * radius * math.pi), 3), dtype='uint8')
    w, h = black_img.shape[0:2]
    for row in range(0, black_img.shape[0]):
        for col in range(0, black_img.shape[1]):
            theta = math.pi * 2 / black_img.shape[1] * (col + 1)
            rho = radius - black_img.shape[0] + row - 1
            p_x = int(circle_center[0] + rho * math.sin(math.pi * 2 - theta) + 0.5) - 1
            p_y = int(circle_center[1] - rho * math.cos(math.pi * 2 - theta) + 0.5) - 1
            black_img[w-row-1, h-col-1, :] = img[p_y, p_x, :]

    black_img_copy = np.copy(black_img)
    SplicingImg = np.concatenate([black_img, black_img_copy], axis=1)
    # cv2.imshow('bk', black_img)
    # cv2.waitKey()
    # cv2.imwrite('bk1.jpg', black_img)
    return SplicingImg


@app.route('/pred_img', methods=["POST"])
def pred_obj():
    Res = {}
    upload_file = request.files['image']

    # res = text_predict(img_path, model_type='cnocr')
    try:
        upload_xml = request.files['xml']
        if upload_xml and allowed_xml(upload_xml.filename):
            filename = upload_xml.filename
            savepath = os.path.join(app.config['UPLOADED_XML_DEST'], filename)
            upload_xml.save(savepath)
            xml_result = GetAnnotBoxLoc(savepath)
            json_res = json.dumps(xml_result, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
            save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'], filename[:-4] + '.json')
            with open(save_name_json, 'w') as file_obj:
                json.dump(json_res, file_obj, cls=NpEncoder)
    except:
        pass

    if upload_file and allowed_file(upload_file.filename) and len(upload_file.filename.split('_')[0]) == 1:
        filename = upload_file.filename
        savepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        upload_file.save(savepath)
        txt_savepath = os.path.join(app.config['UPLOADED_TXT_DEST'], 'test.txt')
        with open(txt_savepath, "a") as file:  # ”w"代表着每次运行都覆盖内容
            file.write(filename.split('.')[0] + "\n")
        s_time = time.time()
        result = inference_detector(model, savepath)

        if isinstance(result, tuple):
            bbox_result, segm_result = result
        else:
            bbox_result, segm_result = result, None
        bboxes = np.vstack(bbox_result)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        num = bboxes.shape[0]
        bboxes_n = []
        img = cv2.imread(savepath)
        for j in range(num):
            bboxes1 = bboxes[j, :]
            bboxes_n.append({"x1": round(bboxes1[0].item(), 2), "y1": round(bboxes1[1].item(), 2),
                             "x2": round(bboxes1[2].item(), 2), "y2": round(bboxes1[3].item(), 2),
                             "confidence": round(bboxes1[4].item(), 3), "label": labels[j]})
            if (labels[j] == 0):
                x1 = int(round(bboxes1[0].item(), 0))
                y1 = int(round(bboxes1[1].item(), 0))
                x2 = int(round(bboxes1[2].item(), 0))
                y2 = int(round(bboxes1[3].item(), 0))
                img_temp = img[y1:y2, x1:x2]
                Radius = min((x1 + x2) / 2.0 - x1, (y1 + y2) / 2.0 - y1)
                out_photo = get_huan_by_circle(img_temp, ((x1 + x2) / 2 - x1, (y1 + y2) / 2 - y1), Radius,
                                               int((x2 - x1) / 4))
                cv2.imwrite('out_{0}.jpg'.format(j), out_photo)

        save_name = os.path.join(app.config['UPLOADED_OUTPUT_DEST'],
                                 str(filename[0:1]) + '_' + str(0) + '_out_' + filename[2:])
        save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'],
                                      str(filename[0:1]) + '_' + str(0) + '_out_' + filename[2:-4] + '.json')
        show_result(savepath, result, model.CLASSES, show=False, out_file=save_name)
        r_time = time.time() - s_time
        if num > 0:
            Res['num'] = num
            Res['bboxes'] = bboxes_n
            Res['predtime'] = round(r_time, 2)
            json_res = json.dumps(Res, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
            # image = open(savepath, mode='rb')
            # resp = Response(image, mimetype="image/jpeg")
        else:
            Res['num'] = 0
            Res['predtime'] = round(r_time, 2)
            json_res = json.dumps(Res, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
        with open(save_name_json, 'w') as file_obj:
            json.dump(Res, file_obj, cls=NpEncoder)

        return json_res
    else:
        Res['state'] = 'please rename!'
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        return json_res


@app.route("/pred_video", methods=["POST"])
def pred_video():
    Res = {}
    upload_file = request.files['image']
    if upload_file and allowed_file_video(upload_file.filename):
        filename = upload_file.filename
        savepath = os.path.join(app.config['UPLOADED_AVI_DEST'], filename)
        print(savepath)
        upload_file.save(savepath)
        s_time = time.time()
        cap = cv2.VideoCapture(savepath)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(savepath[0:-4] + '_out.avi', fourcc, 10.0, (int(cap.get(3)), int(cap.get(4))))
        i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            if i % 6 == 0:
                # cv2.imwrite('123.jpg', frame)
                # res, c, image_write = detect('123.jpg')
                result = inference_detector(model, frame)
                save_name = os.path.join(app.config['UPLOADED_PHOTOS_DEST'],
                                         str(filename[0:1]) + '_' + str(0) + '_out_' + filename[2:])
                # save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'], str(filename[0:1]) + '_' + str(0) + '_out_' + filename[2:-4] + '.json')
                show_result(frame, result, model.CLASSES, show=False, out_file='123.jpg')
                image_read = cv2.imread('123.jpg')
                out.write(image_read)
            i += 1
        cost_time = time.time() - s_time
        cap.release()
        out.release()
        # cv2.destroyAllWindows()
        Res['state'] = 'ok'
        Res['cost_time'] = cost_time
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        print("res:", json_res)
        return json_res
    else:
        Res['state'] = 'developing!'
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        return json_res


@app.route('/clear_output', methods=["post"])
def clear_output():
    import shutil

    json_filepath = app.config['UPLOADED_JSON_DEST']
    video_filepath = app.config['UPLOADED_AVI_DEST']
    photo_filepath = app.config['UPLOADED_PHOTOS_DEST']
    output_filepath = app.config['UPLOADED_OUTPUT_DEST']
    xml_filepath = app.config['UPLOADED_XML_DEST']
    try:
        shutil.rmtree(video_filepath)
        os.mkdir(video_filepath)
        shutil.rmtree(photo_filepath)
        os.mkdir(photo_filepath)
        shutil.rmtree(json_filepath)
        os.mkdir(json_filepath)
        shutil.rmtree(output_filepath)
        os.mkdir(output_filepath)
        shutil.rmtree(xml_filepath)
        os.mkdir(xml_filepath)
        return "clear successfully!"
    except:
        return "clear unsuccessfully!"


@app.route('/manage_jpg')
def manage_file():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    files_list_output = os.listdir(app.config['UPLOADED_OUTPUT_DEST'])
    files_list_dir = []
    for i in files_list:
        if ('_out_' not in i and len(i.split('_')[0]) == 1):
            files_list_dir.append(i)

    for i in files_list_output:
        num = i.split('_')
        if ('_out_' in i and len(num[0]) == 1 and len(num[1]) == 1):
            files_list_map[re.sub('\d_out_', '', i)] = i
    return render_template('manage_jpg.html', files_list=files_list_dir, files_map=files_list_map)


@app.route('/manage_avi')
def manage_file1():
    files_list = os.listdir(app.config['UPLOADED_AVI_DEST'])
    files_list_dir = []
    for i in files_list:
        if '_out' not in i:
            files_list_dir.append(i)

    for i in files_list:
        num = i.split('_')
        if '_out' in i:
            files_list_map[i[:-8] + '.mp4'] = i
    return render_template('manage_avi.html', files_list=files_list_dir, files_map=files_list_map)


@app.route('/open/<filename>')
def open_file(filename):
    file_url = photos.url(filename)
    return render_template('browser.html', file_url=file_url)


@app.route('/open1/<filename>')
def open_file1(filename):
    file_url = avis.url(filename)
    return render_template('browser.html', file_url=file_url)


def Merge(dict1, dict2):
    return (dict2.update(dict1))


@app.route('/open2/<filename>')
def open_file2(filename):
    if (files_list_map.get(filename, 0) != 0 and files_list_map.get(filename, 0) != ''):
        name1, name2 = {}, {}
        file_url = outputs.url(files_list_map[filename])
        save_pre_json = os.path.join(app.config['UPLOADED_JSON_DEST'], filename[0:-4] + '.json')
        save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'], files_list_map[filename][0:-4] + '.json')
        if os.path.exists(save_pre_json):
            with open(save_pre_json) as file_obj:
                name1 = eval(json.load(file_obj))
        if name1.get("num_pre", 0) == 0:
            name1['num_pre'] = 0
        with open(save_name_json) as file_obj:
            name2 = json.load(file_obj)
        name = dict(name2, **name1)
        return render_template('browser_addjson.html', file_url=file_url, **name)
    else:
        return render_template('browser1.html', file_url="", state='no exist photo')


@app.route('/delete/<filename>')
def delete_file(filename):
    file_path = photos.path(filename)
    os.remove(file_path)
    try:
        file_path_output = outputs.path(files_list_map[filename])
        save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'], files_list_map[filename][0:-4] + '.json')
        files_list_map[filename] = 0
        os.remove(file_path_output)
        os.remove(save_name_json)
    except:
        pass
    return redirect(url_for('manage_file'))


@app.route('/delete1/<filename>')
def delete_file1(filename):
    file_path = avis.path(filename)
    os.remove(file_path)
    return redirect(url_for('manage_file1'))


@app.route('/delete2/<filename>')
def delete_file2(filename):
    if (files_list_map.get(filename, 0) != 0):
        try:
            file_path = outputs.path(files_list_map[filename])
            save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'], files_list_map[filename][0:-4] + '.json')
            files_list_map[filename] = 0
            os.remove(file_path)
            os.remove(save_name_json)
        except:
            pass
    return redirect(url_for('manage_file'))


@app.route('/delete3/<filename>')
def delete_file3(filename):
    if (files_list_map.get(filename, 0) != 0):
        file_path = avis.path(files_list_map[filename])
        files_list_map[filename] = 0
        os.remove(file_path)
        return redirect(url_for('manage_file1'))
    else:
        return redirect(url_for('manage_file1'))


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
        file_path = os.path.join(app.config['UPLOADED_OUTPUT_DEST'])
        return send_from_directory(file_path, files_list_map[filename], as_attachment=True)
    else:
        return render_template('browser1.html', file_url="", state='no exist photo')


@app.route('/download3/<filename>')
def download_file3(filename):
    if (files_list_map.get(filename, 0) != 0):
        file_path = os.path.join(app.config['UPLOADED_AVI_DEST'])
        return send_from_directory(file_path, files_list_map[filename], as_attachment=True)
    else:
        return render_template('browser1.html', file_url="", state='no exist avi')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10304, debug=True)
