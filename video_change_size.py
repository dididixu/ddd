# ## -*- coding: utf-8 -*-
import os, cv2


def file_name(file_dir):
    files = os.listdir(file_dir)
    # 找到每一个视频文件
    for file in files:
        file_dir_e = file_dir + "/" + str(file)
        file_n = os.path.splitext(file_dir_e)[0]
        cap = cv2.VideoCapture(file_dir_e)
        frames_speed = round(cap.get(5)+0.5)
        success, _ = cap.read()
        # 重新合成的视频在原文件夹，如果需要分开，可以修改file_n
        videowriter = cv2.VideoWriter(file_n + ".avi", cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), frames_speed,
                                      (640, 640))

        while success:
            success, img1 = cap.read()
            try:
                img = cv2.resize(img1, (640, 640), interpolation=cv2.INTER_LINEAR)
                videowriter.write(img)
            except:
                break


file_name(r"test_video")
