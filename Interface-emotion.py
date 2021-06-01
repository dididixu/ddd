from werkzeug.utils import secure_filename
import os
import cv2
import time
import imutils
import numpy as np
import base64
import json
# import gevent.monkey

# gevent.monkey.patch_all()
from flask import Flask, request
import multiprocessing
# from Emotion import EmotionReg

# GPU按需分配,解决
# import tensorflow as tf
# import keras
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# 进程号
pid = os.getpid()

# 临时变量，避免多进程图片冲撞
temp_i = 0

app = Flask(__name__)
# 文件上传目录
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
# 支持的文件格式
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg'}  # 集合类型


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


def base64_cv2(base64_str):
    imgString = base64.b64decode(base64_str)
    nparr = np.fromstring(imgString, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def hello_world():
    return 'InterFace'


@app.route("/pred_multi", methods=["POST"])
def pre_multi():
    upload_file = request.files['image']
    if upload_file and allowed_file(upload_file.filename):
        filename = secure_filename(upload_file.filename)
        savepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'],
                                str(pid) + "_" + str(temp_i) + "_" + filename)
        upload_file.save(savepath)
        frame = cv2.imread(savepath)
        # res = EmotionReg(frame)
        # os.remove(savepath)
        # json_res = json.dumps(res, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
        return 1
    else:
        respond_error = {}
        respond_error['error_code'] = '1'
        respond_error['result'] = {}
        return respond_error


@app.route("/pred_multi_base64", methods=["GET", "POST"])
def pred_muilt_base64():
    try:
        file = str(request.form['data'])
        data = base64.b64decode(file)
        savepath = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'],
                                str(pid) + "_" + str(temp_i) + "_image.jpg")
        print(savepath)
        with open(savepath, 'wb') as f:
            f.write(data)
        # res = EmotionReg(savepath)
        os.remove(savepath)
        # json_res = json.dumps(res, sort_keys=False, separators=(',', ':'), ensure_ascii=False, cls=NpEncoder)
        return "1"
    except Exception as e:
        respond_error = {}
        respond_error['error_code'] = '1'
        respond_error['result'] = {}
        return respond_error


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10501, debug=True, threaded=True)
    # app.run(port=5000, debug=True)
