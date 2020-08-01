#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import os.path
import math
import cv2
import glob



import numpy as np

#读取照片
# pic_path = 'TSRD-Test/'
pic_path = 'tsrd-train/'
pics = os.listdir(pic_path)

train_label = []
train_data = []
label = 0
for i in pics:
    if i[-4:] == '.png':
        filename = pic_path + i
        img = cv2.imread(filename)
        strint = filename[11:14] #train
        strint = filename[10:13] #test
        label = int(strint)
        print(label)
        res1 = cv2.resize(img, (32,32), interpolation=cv2.INTER_AREA)
        train_data.append(res1 / 255.)
        train_label.append(label)
np.save("train_data32.npy", train_data)
# np.save("test_data32.npy", train_data)
np.save("train_label32.npy", train_label)
# np.save("test_label32.npy", train_label)


