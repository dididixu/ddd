from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request
from torch.autograd import Variable as V
import torch as t
from models.resnet import resnet18
from conf import global_settings
import os
import time
import json
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_dropzone import Dropzone

import cv2
import hashlib
import shutil

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


def allowed_file_video(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS_VIDEO']


def predict(image):
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    trans = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(global_settings.CIFAR100_TRAIN_MEAN, global_settings.CIFAR100_TRAIN_STD)
    ])

    # 读入图片
    # img = Image.open('test3.jpg')
    img = Image.open(image)
    img_deal = cv2.imread(image)
    input = trans(img)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
    img = input.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]
    print("load model......")
    model = resnet18().to(device)  # 导入网络模型
    model.eval()
    model.load_state_dict(t.load('./resnet18-190-regular.pth', map_location='cpu'))  # 加载训练好的模型文件
    print("load model completed.")

    print("inference......")
    input = V(img.to(device))
    score = model(input)  # 将图片输入网络得到输出
    probability = t.nn.functional.softmax(score, dim=1)
    max_value, index = t.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
    # print(index.item())
    a = index.item()
    if a == 0:
        print("黑色")
        b = "黑色"
        state_a = 'black'
    elif a == 1:
        print("蓝色")
        b = "蓝色"
        state_a = 'blue'
    elif a == 2:
        print("咖啡色")
        b = "咖啡色"
        state_a = 'brown'
    elif a == 3:
        print("绿色")
        b = "绿色"
        state_a = 'green'
    elif a == 4:
        print("灰色")
        b = "灰色"
        state_a = 'gray'
    elif a == 5:
        print("紫色")
        b = "紫色"
        state_a = 'purple'
    elif a == 6:
        print("红色")
        b = "红色"
        state_a = 'red'
    elif a == 7:
        print("白色")
        b = "白色"
        state_a = 'white'
    elif a == 8:
        print("黄色")
        b = "黄色"
        state_a = 'yellow'

    cv2.putText(img_deal, '{0} {1:.2f}'.format(state_a, max_value.item()),
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 0), 5,
                cv2.LINE_AA)
    return b, a, img_deal


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route("/pred_img", methods=["POST"])
def pred_ocr():
    Res = {}
    upload_file = request.files['image']
    if upload_file and allowed_file(upload_file.filename) and len(upload_file.filename.split('_')[0]) == 1:
        filename = upload_file.filename
        savepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        print(savepath)
        upload_file.save(savepath)
        s_time = time.time()
        res, c, image_write = predict(savepath)
        cost_time = time.time() - s_time
        Res['colour'] = res
        Res['cost_time'] = cost_time
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        print("res:", json_res)
        save_name = os.path.join(app.config['UPLOADED_PHOTOS_DEST'],
                                 str(filename[0:1]) + '_' + str(c) + '_out_' + filename[2:])
        # shutil.copy(savepath, save_name)
        cv2.imwrite(save_name, image_write)
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
            if i % 3 == 0:
                cv2.imwrite('123.jpg', frame)
                res, c, image_write = predict('123.jpg')
                out.write(image_write)
            i += 1
        cost_time = time.time() - s_time
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        Res['state'] = 'ok'
        Res['cost_time'] = cost_time
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        print("res:", json_res)
        return json_res
    else:
        Res['state'] = 'please rename!'
        json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        return json_res


@app.route('/clear_output', methods=["post"])
def clear_output():
    video_filepath = app.config['UPLOADED_AVI_DEST']
    photo_filepath = app.config['UPLOADED_PHOTOS_DEST']
    try:
        shutil.rmtree(video_filepath)
        os.mkdir(video_filepath)
        shutil.rmtree(photo_filepath)
        os.mkdir(photo_filepath)
        return "clear successfully!"
    except:
        return "clear unsuccessfully!"


files_list = []
files_list_map = {}
import re


@app.route('/')
def manage_file_index():
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


@app.route('/upload', methods=['GET', 'POST'])
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


@app.route('/pred_statistics', methods=["post"])
def pred_statistics():
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

    result = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    from sklearn.metrics import confusion_matrix, classification_report
    y_true = []
    y_pred = []
    target_names = ['黑色', '蓝色', '咖啡色', '绿色', '灰色', '紫色', '红色', '白色', '黄色']
    for files in result:
        file = files_list_map.get(files, 0)
        if file == 0:
            continue
        label, predict = file.split("_")[0], file.split("_")[1]
        if (len(label) == 1 and len(predict) == 1):
            y_true.append(int(label))
            y_pred.append(int(predict))
    print(y_true, y_pred)
    # s = confusion_matrix(y_true, y_pred)
    try:
        return classification_report(y_true, y_pred, target_names=target_names)
    except:
        return 'statistics error!,please add more photos'


@app.route('/manage_jpg')
def manage_file():
    files_list = os.listdir(app.config['UPLOADED_PHOTOS_DEST'])
    files_list_dir = []
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


@app.route('/open2/<filename>')
def open_file2(filename):
    if (files_list_map.get(filename, 0) != 0 and files_list_map.get(filename, 0) != ''):
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
        files_list_map[filename] = 0
        os.remove(file_path)
        return redirect(url_for('manage_file'))
    else:
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
        file_path = os.path.join(app.config['UPLOADED_PHOTOS_DEST'])
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
    app.run(host='0.0.0.0', port=10302, debug=True, processes=True)
