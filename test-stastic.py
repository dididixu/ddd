#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: test-stastic.py
@time: 2021/2/2 15:39
@desc: 
"""
import time
import threading
a = 0

def fun():
    global a
    while True:
        time.sleep(1)
        a += 1

class A():
    def __init__(self,id):
        A.get(id)

    @staticmethod
    def get(id):
        A.id = id

    @staticmethod
    def show():
        global a
        while True:
            time.sleep(1)
            print(a)

if __name__ == '__main__':
    c = A(0)
    t1 = threading.Thread(target=fun, daemon=True)
    t1.start()
    c.show()