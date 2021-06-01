# -*- coding:utf-8 -*-
import base64
import cv2
import requests

def post(img_path):  # 转为二进制格式
    img = cv2.imread(img_path)
    url = 'http://192.168.0.68:10200/pred_single_base64'
    post_data = {
        'data': base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()
    }
    print(post_data)
    r = requests.post(url, data=post_data)
    print(r.text)


if __name__ == '__main__':
    post('222222.jpg')

f = h5py.File("model_weights.h5",'r+')
for k in [.25, .50, .60, .70, .80, .90, .95, .97, .99]:
    ranks = {}
    for l in list(f['model_weights'])[:-1]:
        data = f['model_weights'][l][l]['kernel:0']
        w = np.array(data)
        ranks[l]=(rankdata(np.abs(w),method= 'dense')—1).astype(int).reshape(w.shape)
        lower_bound_rank = np.ceil(np.max(ranks[l])*k).astype(int)
        ranks[l][ranks[l]<=lower_bound_rank] = 0
        ranks[l][ranks[l]>lower_bound_rank] = 1
        w = w*ranks[l]
        data[…] = w

f = h5py.File("model_weights.h5",'r+')
for k in [.25, .50, .60, .70, .80, .90, .95, .97, .99]:
    ranks = {}
    for l in list(f['model_weights'])[:-1]:
        data = f['model_weights'][l][l]['kernel:0']
        w = np.array(data)
        norm = LA.norm(w,axis=0)
        norm = np.tile(norm,(w.shape[0],1))
        ranks[l] = (rankdata(norm,method='dense')—1).astype(int).reshape(norm.shape)
        lower_bound_rank = np.ceil(np.max(ranks[l])*k).astype(int)
        ranks[l][ranks[l]<=lower_bound_rank] = 0
        ranks[l][ranks[l]>lower_bound_rank] = 1
        w = w*ranks[l]
        data[…] = w