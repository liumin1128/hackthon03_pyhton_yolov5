import cv2
import numpy as np

print("hello wrold")


def loadcv2dnnNetONNX(onnx_path):
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print('load successful')
    return net


def pad2square_cv2(image):
    h, w, c = image.shape
    dim_diff = np.abs(h-w)
    pad1, pad2 = dim_diff//2, dim_diff-dim_diff//2
    if h <= w:
        image = cv2.copyMakeBorder(image, pad1, pad2, 0, 0,
                                   cv2.BORDER_CONSTANT, value=0)
    else:
        image = cv2.copyMakeBorder(image, 0, 0, pad1, pad2,
                                   cv2.BORDER_CONSTANT, value=0)
    return image


def preprocess4cv2dnn(image_path):
    image = cv2.imread(image_path)

    image = pad2square_cv2(image)

    h, w = image.shape[:2]

    # 干嘛要水平翻转？
    # image = cv2.flip(image, 1)

    blobImage = cv2.dnn.blobFromImage(image, 1.0 / 255.0, (416, 416),
                                      None, True, False)

    return blobImage, image


# net = cv2.dnn.readNetFromONNX("../yolov5/weights/yolov5s.onnx")

# img = cv2.imread("./images/1.jpg")
# h, w = img.shape[:2]  # 获取图像的高和宽
# cv2.imshow("Origin", img)  # 显示原始图像

# blob = cv2.dnn.blobFromImage(
#     img, 1 / 255.0, (w, h), [0, 0, 0], swapRB=True, crop=False)

def testcv2nnNet(img_path, model_path):
    blobImage, image = preprocess4cv2dnn(img_path)
    h, w = image.shape[:2]

    net = loadcv2dnnNetONNX(model_path)

    outNames = net.getUnconnectedOutLayersNames()
    net.setInput(blobImage)
    outs = net.forward(outNames)

    anchors = [
        [(116, 90), (156, 198), (373, 326)],  # 13*13 上预测最大的
        [(30, 61), (62, 45), (59, 119)],  # 26*26 上预测次大的
        [(10, 13), (16, 30), (33, 23)],  # 13*13 上预测最小的
    ]

    yolo1 = YOLO_NP(anchors[0], 2, 416)
    yolo2 = YOLO_NP(anchors[1], 2, 416)
    yolo3 = YOLO_NP(anchors[2], 2, 416)

    yolo_output1 = yolo1(outs[0])
    yolo_output2 = yolo2(outs[1])
    yolo_output3 = yolo3(outs[2])

    detections = np.concatenate([yolo_output1, yolo_output2, yolo_output3], 1)

    detections = non_max_suppression_np(detections, 0.5, 0.4)[0]

    print('detect res ', len(detections))
    if detections is not None:
        detections = rescale_boxes(detections, 416, (h, w))

        # 显示就自己写吧…

        cv2.imshow("Origin", image)  # 显示原始图像


blobImage, image = testcv2nnNet(
    "./images/1.jpg", "../yolov5/weights/yolov5s.onnx")
h, w = image.shape[:2]

cv2.waitKey(0)
cv2.destroyAllWindows()
