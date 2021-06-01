from flask import Flask, request
import Single_facial_expression
import Multi_facial_expression
from werkzeug.utils import secure_filename
import os
import cv2
import time
import imutils
import numpy as np
import base64
import json
import gevent.monkey

gevent.monkey.patch_all()
import multiprocessing

app = Flask(__name__)
# 文件上传目录
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg'}  # 集合类型
w = 320
h = 240


def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


# 判断文件名是否是我们支持的格式
def load_model():
    global detector_dlib, emotion_model, age_gander_model
    detector_dlib = Single_facial_expression.Init_detector_dlib()
    emotion_model = Single_facial_expression.Init_emotion_classifier()
    age_gander_model = Single_facial_expression.Init_Age_Gnender_model()
    # 初始化加载模型之后，就随便生成一个向量让 model 执行一次 predict 函数，否则会报错。
    x = np.empty((1, 64, 64, 3))
    age_gander_model.predict(x)
    emotion_model.eval()

load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def hello_world():
    return 'InterFace'


@app.route("/pred_single", methods=["POST"])
def pred_single():
    upload_file = request.files['image']
    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)
        savepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        upload_file.save(savepath)
        # time.sleep(0.5)
        frame = cv2.imread(savepath)
        frame = imutils.resize(frame, width=w, height=h)
        res = Single_facial_expression.GetEmotion(frame, detector_dlib, age_gander_model, emotion_model)
        os.remove(savepath)
        return res
    else:
        return None


@app.route("/pred_multi", methods=["POST"])
def pre_multi():
    upload_file = request.files['image']
    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)
        savepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
        upload_file.save(savepath)
        # time.sleep(0.5)
        frame = cv2.imread(savepath)
        frame = imutils.resize(frame, width=w, height=h)
        res = Multi_facial_expression.GetEmotion(frame, detector_dlib, age_gander_model, emotion_model)
        os.remove(savepath)
        return res
    else:
        return None


@app.route("/pred_single_base64", methods=["GET", "POST"])
def pred_single_base64():
    try:
        # info = request.get_data(as_text=True)
        # start_time = time.time()
        file = request.form['data']
        frame = base64_cv2(file)
        # imgdata = base64.b64decode(file)
        # with open('temp_img.jpg', 'wb') as file:
        #     file.write(imgdata)
        # frame = cv2.imread('temp_img.jpg')
        frame = imutils.resize(frame, width=w, height=h)
        res = Single_facial_expression.GetEmotion(frame, detector_dlib, age_gander_model, emotion_model)
        # os.remove('./temp_img.jpg')
        # print("cost time: ", time.time() - start_time)
        return res
    except Exception as e:
        respond_error = {}
        respond_error['error_code'] = '1'
        respond_error['result'] = {}
        respond_error['result']['face_list'] = []
        res_temp = {}
        res_temp['face_token'] = None
        res_temp['emotion'] = {}
        res_temp['emotion']['type'] = None
        res_temp['emotion']['probability'] = None
        res_temp['location'] = {}
        res_temp['location']['top'] = None
        res_temp['location']['bottom'] = None
        res_temp['location']['left'] = None
        res_temp['location']['right'] = None
        res_temp['location']['width'] = None
        res_temp['location']['height'] = None
        res_temp['expression'] = {}
        respond_error['result']['face_list'].append(res_temp)
        return respond_error
    # start_time = time.time()
    # if info:
    #     try:
    #         json_info = json.loads(info)
    #         with open('temp_img.jpg', 'wb') as file:
    #             pixel = base64.b64decode(json_info["data"])
    #             file.write(pixel)
    #         frame = cv2.imread('temp_img.jpg')
    #         # frame = imutils.resize(frame, width=640, height=480)
    #         cv2.imwrite('te.jpg', frame)
    #         res = Single_facial_expression.GetEmotion(frame, detector_dlib, age_gander_model, emotion_model)
    #         # os.remove('./temp_img.jpg')
    #         print("cost time: ", time.time() - start_time)
    #         return res
    #     except Exception as e:
    #         respond_error = {}
    #         respond_error['error_code'] = '1'
    #         respond_error['result'] = {}
    #         respond_error['result']['face_list'] = []
    #         res_temp = {}
    #         res_temp['face_token'] = None
    #         res_temp['emotion'] = {}
    #         res_temp['emotion']['type'] = None
    #         res_temp['emotion']['probability'] = None
    #         res_temp['location'] = {}
    #         res_temp['location']['top'] = None
    #         res_temp['location']['bottom'] = None
    #         res_temp['location']['left'] = None
    #         res_temp['location']['right'] = None
    #         res_temp['location']['width'] = None
    #         res_temp['location']['height'] = None
    #         res_temp['expression'] = {}
    #         respond_error['result']['face_list'].append(res_temp)
    #         return respond_error
    # else:
    #     return 'cant get the info from your post request'


@app.route("/pred_multi_base64", methods=["GET", "POST"])
def pred_muilt_base64():
    # info=request.form['info']
    # info=request.get_data(as_text=True)
    #     # with open('temp_img.jpg','wb') as file:
    #     #     pixel = base64.b64decode(info)
    #     #     file.write(pixel)
    #     # frame=cv2.imread('./temp_img.jpg')
    #     # frame=imutils.resize(frame,width=640,height=480)
    #     # res = Multi_facial_expression.GetEmotion(frame, detector_dlib, age_gander_model, emotion_model)
    #     #
    #     # os.remove('./temp_img.jpg')
    try:
        file = request.form['data']
        frame = base64_cv2(file)
        # imgdata = base64.b64decode(file)
        # with open('temp_img.jpg', 'wb') as file:
        #     file.write(imgdata)
        # frame = cv2.imread('temp_img.jpg')
        frame = imutils.resize(frame, width=w, height=h)
        res = Multi_facial_expression.GetEmotion(frame, detector_dlib, age_gander_model, emotion_model)
        # os.remove('./temp_img.jpg')
        # print("cost time: ", time.time() - start_time)
        return res
    except Exception as e:
        respond_error = {}
        respond_error['error_code'] = '1'
        respond_error['result'] = {}
        respond_error['result']['face_list'] = []
        res_temp = {}
        res_temp['face_token'] = None
        res_temp['emotion'] = {}
        res_temp['emotion']['type'] = None
        res_temp['emotion']['probability'] = None
        res_temp['location'] = {}
        res_temp['location']['top'] = None
        res_temp['location']['bottom'] = None
        res_temp['location']['left'] = None
        res_temp['location']['right'] = None
        res_temp['location']['width'] = None
        res_temp['location']['height'] = None
        res_temp['expression'] = {}
        respond_error['result']['face_list'].append(res_temp)
        return respond_error


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=10200, debug=True, threaded=True)
    # app.run(port=5000, debug=True)
