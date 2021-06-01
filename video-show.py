import cv2
import os
import glob
import time

root_dir = r"D:\test-video"
path = os.path.join(root_dir, r"police.mp4")
rtsp = "rtsp://admin:123456@192.168.0.50:554"
cap = cv2.VideoCapture(path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# path_file_number = glob.glob(pathname="./photo-train/*.jpg")
fps = cap.get(cv2.CAP_PROP_FPS)
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(fps, size)
out = cv2.VideoWriter('output-1028.avi', fourcc, 25, size)
# i = len(path_file_number)
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    # time.sleep(0.04)
    out.write(frame)
    # cv2.imshow('frame', frame)
    # c = cv2.waitKey(1)
    # if c == ord('s'):
    #     cv2.imwrite('./photo-train/train-' + str("%06d" % i) + '.jpg', frame)  # 存储为图像
    #     print("successful save!")
    #     i = i + 1
    # if (c == 27 or c == ord('q')):
    #     break

cap.release()
out.release()
cv2.destroyAllWindows()
