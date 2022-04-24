import os
from re import template
from flask import Flask, request
from flask_cors import CORS
import uuid
import numpy as np
import cv2


def format_yolov5(frame):

    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result


def auto_crop(name):

    net = cv2.dnn.readNet('config_files/yolov5s.onnx')

    image = cv2.imread('./static/'+name)
    input_image = format_yolov5(image)  # making the image square
    blob = cv2.dnn.blobFromImage(input_image, 1/255.0, (640, 640), swapRB=True)
    net.setInput(blob)
    predictions = net.forward()

    class_ids = []
    confidences = []
    boxes = []
    crop_info = []
    crop_name = ""

    output_data = predictions[0]

    image_width, image_height, _ = input_image.shape
    x_factor = image_width / 640
    y_factor = image_height / 640

    for r in range(25200):
        row = output_data[r]
        confidence = row[4]

        if confidence >= 0.4:

            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(
                ), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    class_list = []
    with open("config_files/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    for i in range(len(result_class_ids)):

        box = result_boxes[i]
        class_id = result_class_ids[i]

        if class_id != 45:
            continue

        crop_info = box
        x, y, w, h = box

        crop = image[y:y+h, x:x+w]
        # cv2.imshow('crop', crop)

        crop_name = "crop_" + name
        cv2.imwrite("./static/"+crop_name, crop)

        cv2.rectangle(image, box, (0, 255, 255), 2)
        cv2.rectangle(image, (box[0], box[1] - 20),
                      (box[0] + box[2], box[1]), (0, 255, 255), -1)
        cv2.putText(image, class_list[class_id], (box[0],
                    box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

    # cv2.imwrite("./image_output/1.jpg", image)
    # cv2.imshow("output", image)
    # cv2.waitKey()
    return crop_name, crop_info


app = Flask(__name__, static_url_path='/static')
cors = CORS(app)

@app.route('/upload', methods=['POST'])
def upload():
    # 上传文件
    file = request.files.get("file")
    filename = str(uuid.uuid4()) + "_" + file.filename
    filepath = os.path.join("static", filename)
    file.save(filepath)
    url = "http://localhost:3101/static/"+filename

    crop_name, crop_info = auto_crop(filename)
    crop_url = "http://localhost:3101/static/"+crop_name

    x, y, w, h = crop_info

    template = '{"status":%d,"url":"%s","crop_url":"%s","crop_info":{"x":%d,"y":%d,"w":%d,"h":%d}}'

    return template%(200, url, crop_url, x, y, w, h)


if __name__ == '__main__':

    app.run(host='0.0.0.0', port=3101,  debug=True)
