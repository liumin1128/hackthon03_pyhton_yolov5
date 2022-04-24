# 轮廓提取

import cv2 as cv
import numpy as np


def edge_demo(image):
    blurred = cv.GaussianBlur(image, (3, 3), 0)  # 高斯模糊
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)  # 灰路图像
    xgrad = cv.Sobel(gray, cv.CV_16SC1, 1, 0)  # xGrodient
    ygrad = cv.Sobel(gray, cv.CV_16SC1, 0, 1)  # yGrodient
    edge_output = cv.Canny(xgrad, ygrad, 100, 150)  # edge
    cv.imshow("Canny Edge", edge_output)

    # #  彩色边缘
    # dst = cv.bitwise_and(image, image, mask=edge_output)
    # cv.imshow("Color Edge", dst)


src = cv.imread("./1.png")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)
edge_demo(src)
cv.waitKey(0)


cv.destroyAllWindows()
