#!/usr/bin/python3
# -*- coding:utf-8 -*-
"""
@author: xcd
@file: test-video-connect.py
@time: 2021/2/22 13:19
@desc: 
"""
import cv2

import time
import multiprocessing as mp

import queue
import threading
import subprocess as sp

"""
Source: Yonv1943 2018-06-17
https://github.com/Yonv1943/Python/tree/master/Demo
"""


class Live(object):
    def __init__(self):
        self.frame_queue = queue.Queue()
        self.command = ""
        # 自行设置
        self.rtmpUrl = "rtmp://192.168.0.132/live/402880f076ade2da0176b27950b10023"
        self.camera_path = "rtmp://192.168.0.134/live/test11"

    def read_frame(self):
        print("开启推流")
        cap = cv2.VideoCapture(self.camera_path)

        # Get video information
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(fps, width, height)

        # ffmpeg command
        self.command = ['ffmpeg',
                        '-y',
                        '-f', 'rawvideo',
                        '-vcodec', 'rawvideo',
                        '-pix_fmt', 'bgr24',
                        '-s', "{}x{}".format(width, height),
                        '-r', str(fps),
                        '-i', '-',
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'ultrafast',
                        '-f', 'flv',
                        self.rtmpUrl]

        # read webcamera
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                print("Opening camera is failed")
                break

            # put frame into queue
            self.frame_queue.put(frame)

    def push_frame(self):
        # 防止多线程时 command 未被设置
        while True:
            if len(self.command) > 0:
                # 管道配置
                p = sp.Popen(self.command, stdin=sp.PIPE)
                break

        while True:
            if self.frame_queue.empty() != True:
                frame = self.frame_queue.get()
                # process frame
                # 你处理图片的代码
                # write to pipe
                p.stdin.write(frame.tostring())

    def run(self):
        threads = [
            threading.Thread(target=Live.read_frame, args=(self,)),
            threading.Thread(target=Live.push_frame, args=(self,))
        ]
        [thread.setDaemon(True) for thread in threads]
        [thread.start() for thread in threads]


if __name__ == '__main__':
    # run_multi_camera()
    live = Live()
    live.run()
