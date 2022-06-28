import cv2
import numpy as np
from performance.bounding_boxes import BoundingBoxes
from detections.yolo_predictions import YoloPredictions
from helpers.frame import Frame

save_path = '/home/ellentuane/Documents/IC/Flight Security/detections/extracted_bbox'
video_path = '/home/ellentuane/Documents/IC/Flight Security/detections/video/test.mp4'
classes_path = '/home/ellentuane/Documents/IC/Flight Security/detections/classes/coco.names'
cfg_path = '/home/ellentuane/Documents/IC/Flight Security/detections/cfg/yolov4-tiny.cfg'
weight_path = '/home/ellentuane/Documents/IC/Flight Security/detections/weights/yolov4-tiny.weights'

# .names files with the object's names
labels = open(classes_path).read().strip().split('\n')

# Random colors for each object category
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# yolo weights and cfg configuration files
net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

use_gpu = 0
if use_gpu == 1:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Obter o nome das categorias
layer_names = YoloPredictions.layer_name(net)

cap = cv2.VideoCapture(video_path)
video_name = video_path.split("/")[-1]
video_name = video_name.split(".")[0]

stop = 0
i = 0
while True:
    if stop == 0:
        ret, frame = cap.read()
        if ret:
            #net, layer_names, image, confidence, threshold, net_height, net_width
            boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, frame,
                                                                                 0.01, 0.03, 960, 960)

            idx_index = -1
            for class_id, score, bbox, idx in zip(classIDs, confidences, boxes, idxs):
                class_name = labels[class_id]
                if class_name == 'person':
                    idxs = np.delete(idxs, idx_index)
            idx_index += 1

            frame = BoundingBoxes.draw_bounding_boxes(frame, labels, boxes, confidences, classIDs, idxs, colors)
            frame = Frame.image_center(frame)

            cv2.imshow('frame', frame)

        else:
            print('Video has ended, failed or wrong path was given, try a different video format!')
            break
        i += 1

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        stop = not stop
    if key == ord('q'):
        break


cv2.destroyAllWindows()