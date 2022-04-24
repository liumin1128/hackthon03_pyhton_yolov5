from PIL import Image
from yolo import YOLO
import os
import cv2
import numpy as np
yolo = YOLO()


'''
yolo抠图，截取目标
'''
j = 0
# 预测图片所在路径
path = './images'
imgdir = os.listdir(path)
for dir in imgdir:
    img_path = os.path.join(path, dir)
    image = Image.open(img_path)
    # print(image)
    crop_image = cv2.imread(img_path)
    # print(crop_image[0])
    boxes = yolo.detect_image(image)
    # print(boxes)

    top = boxes[0][0]
    left = boxes[0][1]
    bottom = boxes[0][2]
    right = boxes[0][3]

    top = top - 5
    left = left - 5
    bottom = bottom + 5
    right = right + 5

    # 左上角点的坐标
    top = int(max(0, np.floor(top + 0.5).astype('int32')))
    left = int(max(0, np.floor(left + 0.5).astype('int32')))
    # 右下角点的坐标
    bottom = int(min(np.shape(image)[0], np.floor(
        bottom + 0.5).astype('int32')))
    right = int(min(np.shape(image)[1], np.floor(right + 0.5).astype('int32')))

    croped_region = crop_image[top:bottom, left:right]

    # 裁剪图片存放目录
    baocun = r'./output'
    save_path = os.path.join(baocun, str(j) + '.bmp')
    cv2.imwrite(save_path, croped_region)
    j = j + 1
