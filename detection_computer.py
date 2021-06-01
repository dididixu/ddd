import argparse
from sys import platform
import hashlib
from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
from PIL import Image
import torchvision.transforms as transforms
from flask import Flask, request
from torch.autograd import Variable as V

from mod.resnet import resnet18
from conf import global_settings
################################################
# 以下代码都需要复制，模板通用类

import time, json, re, cv2, os, shutil
from flask import Flask, render_template, redirect, url_for, request, send_from_directory
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_dropzone import Dropzone
import numpy as np


# dict->json类型转换失败，自定义一个函数
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


###############################################
def detect(image):
    cc = 2
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
    classify = False
    trans = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(global_settings.CIFAR100_TRAIN_MEAN, global_settings.CIFAR100_TRAIN_STD)
    ])
    # Initialize
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    model2 = resnet18().to(device)  # 导入网络模型
    model2.eval()
    model2.load_state_dict(torch.load('./resnet18-190-regular.pth', map_location='cpu'))  # 加载训练好的模型文件
    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Eval mode
    model.to(device).eval()

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        # print(123)
        dataset = LoadImages(image, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        print("1111", pred)
        image_write = cv2.imread(image)
        for i, det in enumerate(pred):  # detections per image
            p, s, im0 = path, '', im0s
            s += '%gx%g ' % img.shape[2:]  # print string
            b = []
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # work detection

                for *xyxy, conf, cls in det:
                    color = (255, 0, 0)
                    imgcolor = cv2.imread(image)
                    imgcolor = imgcolor[int(xyxy[1].item()):int(xyxy[3].item()),
                               int(xyxy[0].item()):int(xyxy[2].item())]

                    # 左上角 -> 右下角
                    input = Image.fromarray(cv2.cvtColor(imgcolor, cv2.COLOR_BGR2RGB))

                    input = trans(input)  # 这里经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小B
                    img = input.unsqueeze(0)  # 增加一维，输出的img格式为[1,C,H,W]

                    input = V(img.to(device))
                    score = model2(input)  # 将图片输入网络得到输出
                    probability = torch.nn.functional.softmax(score, dim=1)
                    max_value, index = torch.max(probability, 1)  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别
                    if index.item() == 0:
                        a = "检测到一台设备黑屏"
                        state_a = 'warning!'
                        cc = 0
                        color = (0, 0, 255)
                    else:
                        a = "一台设备正常运行"
                        state_a = 'working'
                        cc = 1

                    cv2.rectangle(image_write, (int(xyxy[0].item()), int(xyxy[1].item())),
                                  (int(xyxy[2].item()), int(xyxy[3].item())), color, 2)
                    cv2.putText(image_write, '{0} {1:.2f}'.format(state_a, max_value.item()),
                                (int(xyxy[0].item()), int(xyxy[1].item()) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, color, 1,
                                cv2.LINE_AA)
                    b.append({'x1': int(xyxy[0].item()), 'y1': int(xyxy[0].item()),
                              'x2': int(xyxy[0].item()), 'y2': int(xyxy[0].item()),
                              'confidence': round(max_value.item(), 3), 'label': round(index.item(), 3)})
                    # save_name = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], str(filename[0:1]) + '_' + str(cc) + '_out_' + filename[2:])
                    # img.save(save_name)
                    # cv2.imwrite(save_name, image_write)
        return b, cc, image_write
    # Print time (inference + NMS)
    # print('%sDone. (%.3fs)' % (s, time.time() - t))


###################################################
# 以下代码都需要复制，模板通用类
files_list = []
files_list_map = {}


@app.route('/', methods=['GET', 'POST'])
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


@app.route("/pred_img", methods=["POST"])
def pred_ocr():
    Res = {}
    upload_file = request.files['image']
    if upload_file and allowed_file(upload_file.filename) and len(upload_file.filename.split('_')[0]) == 1:
        filename = upload_file.filename
        savepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        upload_file.save(savepath)
        s_time = time.time()
        res, c, image_write = detect(savepath)
        cost_time = time.time() - s_time
        #######################################################
        if len(res) > 0:
            Res['num'] = len(res)
            Res['bboxes'] = res
            Res['predtime'] = round(cost_time, 3)
            json_res = json.dumps(Res, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
            # image = open(savepath, mode='rb')
            # resp = Response(image, mimetype="image/jpeg")
        else:
            Res['num'] = 0
            Res['predtime'] = round(cost_time, 3)
            json_res = json.dumps(Res, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)

        save_name_json = os.path.join(app.config['UPLOADED_JSON_DEST'],
                                      str(filename[0:1]) + '_' + str(c) + '_out_' + filename[2:-4] + '.json')
        with open(save_name_json, 'w') as file_obj:
            json.dump(Res, file_obj, cls=NpEncoder)
        #######################################################

        # json_res = json.dumps(Res, sort_keys=True, indent=4, separators=(',', ': '), ensure_ascii=False)
        # print("res:", json_res)
        save_name = os.path.join(app.config['UPLOADED_PHOTOS_DEST'],
                                 str(filename[0:1]) + '_' + str(c) + '_out_' + filename[2:])
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
                cv2.imwrite('123.jpg', frame)
                res, c, image_write = detect('123.jpg')
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
    import shutil

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
    target_names = ['黑屏', '正常', '无屏幕']
    for files in result:
        file = files_list_map.get(files, 0)
        if file == 0:
            continue
        label, predict = file.split("_")[0], file.split("_")[1]
        if (len(label) == 1 and len(predict) == 1):
            y_true.append(int(label))
            y_pred.append(int(predict))
    # s = confusion_matrix(y_true, y_pred)
    try:
        return classification_report(y_true, y_pred, target_names=target_names)
    except:
        return 'statistics error!'


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='data/fire.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='weights/best.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='3.jpg', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    # print(opt)

    app.run(host='0.0.0.0', port=10303, debug=True, processes=True)
    # with torch.no_grad():
    #   detect()
