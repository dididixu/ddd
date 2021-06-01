# ## -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.autograd import Variable
import cv2
from torchvision import models, utils, transforms
from PIL import Image
import numpy as np
import os


def load_checkpoint(model, resume):
    if os.path.isfile(resume) != None:
        checkpoint = torch.load(resume)
        # print(model_CKPT)
        print(checkpoint)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(model_CKPT['optimizer'])
    return model, best_prec1

# use_gpu = 0
PATH = "./model_best.pth.tar"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0" if use_gpu else "cpu")

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),  # 尺寸规范
    transforms.ToTensor(),  # 转化为tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

model = models.__dict__["resnext50_32x4d"](pretrained=True).to(device)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(2048, 2)
model = torch.nn.DataParallel(model).to(device)
# model = models.resnext101_32x8d().to(device)
model, prec = load_checkpoint(model, PATH)
# resnetx = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)

model.eval()
print('loading checkpoint!')

img_dir = "dataset-desk/clean"
img = os.listdir(img_dir)
for i in img:
    img_path = os.path.join(img_dir, i)
    # img_path = r"dataset-desk/messy/00000272.jpg"
    img = Image.open(img_path)
    img = transform(img)
    img = img.reshape((1, 3, 224, 224))
    # print(img.shape)

    with torch.no_grad():
        inputs = Variable(img.to(device))
        y_pred = model(inputs)
        smax = nn.Softmax()
        smax_out = smax(y_pred)[0]
        # prob = smax_out.data[0]
        # if smax_out.data[0] > smax_out.data[1]:
        #     prob = 1 - smax_out.data[0]
        # prob = np.around(prob.to(device), decimals=4)
        print(smax_out)

    # res = model(img)
