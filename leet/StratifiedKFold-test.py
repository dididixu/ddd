#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: StratifiedKFold-test.py
@time: 2021/1/26 10:14
@desc: 
"""

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

X = np.array([
    [1, 2, 3, 4],
    [11, 12, 13, 14],
    [21, 22, 23, 24],
    [31, 32, 33, 34],
    [41, 42, 43, 44],
    [51, 52, 53, 54],
    [61, 62, 63, 64],
    [71, 72, 73, 74]
])

y = np.array([1, 1, 1, 1, 1, 1, 0, 0])

sfolder = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
folder = KFold(n_splits=4, random_state=0, shuffle=False)

for train, test in sfolder.split(X, y):
    print(train, test)

print("-------------------------------")
for train, test in folder.split(X, y):
    print(train, test)

for fold, (train_idx, val_idx) in enumerate(sfolder.split(X, y)):
    train_set, val_set = X[train_idx], X[val_idx]
