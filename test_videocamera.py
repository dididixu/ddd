# created by Huang Lu
# 27/08/2016 17:05:45
# Department of EE, Tsinghua Univ.

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while (1):
    # get a frame
    ret, frame = cap.read()
    # show a frame
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

import random


def generator(max):
    number = 1
    while number < max:
        number += 1
        yield number
        # Create as stream generator
        stream = generator(10000)  # Doing Reservoir Sampling from the stream
        k = 5
        reservoir = []
        for i, element in enumerate(stream):
            if i + 1 <= k:
                reservoir.append(element)
            else:
                probability = k / (i + 1)
            if random.random() < probability:  # Select item in stream and remove one of the k items already selected
                reservoir[random.choice(range(0, k))] = element
                print(reservoir)  # [1369, 4108, 9986, 828, 5589]
