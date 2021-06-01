#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: meshwarp.py
@time: 2021/2/3 16:31
@desc: 
"""
import numpy as np
import matplotlib.pyplot as plt

mr = 88
mc = 68

xx = np.arange(mr-1, -1, -1)
yy = np.arange(0, mc, 1)
[Y, X] = np.meshgrid(xx, yy)
ms = np.transpose(np.asarray([X.flatten('F'), Y.flatten('F')]), (1,0))

perturbed_mesh = ms
nv = np.random.randint(20) - 1
for k in range(nv):
    #Choosing one vertex randomly
    vidx = np.random.randint(np.shape(ms)[0])
    vtex = ms[vidx, :]
    #Vector between all vertices and the selected one
    xv  = perturbed_mesh - vtex
    #Random movement
    mv = (np.random.rand(1,2) - 0.5)*20
    hxv = np.zeros((np.shape(xv)[0], np.shape(xv)[1] +1) )
    hxv[:, :-1] = xv
    hmv = np.tile(np.append(mv, 0), (np.shape(xv)[0],1))
    d = np.cross(hxv, hmv)
    d = np.absolute(d[:, 2])
    d = d / (np.linalg.norm(mv, ord=2))
    wt = d

    curve_type = np.random.rand(1)
    if curve_type > 0.3:
        alpha = np.random.rand(1) * 50 + 50
        wt = alpha / (wt + alpha)
    else:
        alpha = np.random.rand(1) + 1
        wt = 1 - (wt / 100 )**alpha
    msmv = mv * np.expand_dims(wt, axis=1)
    perturbed_mesh = perturbed_mesh + msmv

plt.scatter(perturbed_mesh[:, 0], perturbed_mesh[:, 1], c=np.arange(0, mr*mc))
plt.show()