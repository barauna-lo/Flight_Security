import os

import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np

from helpers.frame import Frame
from performance.bounding_boxes import BoundingBoxes
from performance.yolo_predictions import YoloPredictions
from helpers.net_size import change_net_size
from coordinates.geo_coordinates import gsd
from helpers.frame import Frame

flags.DEFINE_string('cfg', './detections/cfg/yolov4.cfg', 'path to cfg file')
flags.DEFINE_integer('size', 1280, 'resize images to')
flags.DEFINE_string('model', 'tiny', 'tiny or yolov4')
flags.DEFINE_string('weights', './detections/weights/yolov4.weights', 'path to weights file')
flags.DEFINE_string('data_path', './detections/tste', 'path to frames or video')
flags.DEFINE_string('output', './detections/extracted_bbox', 'path to output bboxes')
flags.DEFINE_string('classes', './detections/classes/coco.names', 'path to classes name video')
flags.DEFINE_string('data_type', 'frame', 'set video or frame')


def main(_argv):
    # .names files with the object's names
    labels = open(FLAGS.classes).read().strip().split('\n')

    # Random colors for each object category
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # yolo weights and cfg configuration files
    # coco weights and cfg files are set as default
    # in case tiny weight and cfg are activated files path will change
    if FLAGS.model == 'tiny':
        # yolo weights and cfg configuration files
        cfg_path = './detections/cfg/yolov4-tiny.cfg'
        weight_path = './detections/weights/yolov4-tiny.weights'

        # changing net size in cfg file
        if FLAGS.size != 416:
            change_net_size(FLAGS.size, cfg_path)

        net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)
    else:
        # changing net size in cfg file
        if FLAGS.size != 416:
            change_net_size(FLAGS.size, FLAGS.cfg)

        # in case a new path for new weights and cfg is added, it will be executed in this step, else it'll be executing yolov4 weights
        net = cv2.dnn.readNetFromDarknet(FLAGS.cfg, FLAGS.weights)

    # Obter o nome das categorias
    layer_names = YoloPredictions.layer_name(net)

    if FLAGS.data_type == 'frame':
        i = 0
        data = []
        image_name_list = []
        # loading images
        for frame in os.listdir(FLAGS.data_path):

            # remove .jpg or any image type from image name
            image_name = frame.split(".")[0]
            image_name_list.append(image_name)

            image = cv2.imread(os.path.join(FLAGS.data_path, frame))

            if not image is None:

                # net, layer_names, image, confidence, threshold, net_height, net_width
                boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, image,
                                                                                     0.005, 0.005, FLAGS.size, FLAGS.size)

                print(image_name, ':')
                idx_index = 0
                for idx in idxs:

                    class_name = labels[classIDs[idx]]
                    if class_name == "person":
                        print(class_name, round(confidences[idx] * 100), "%")
                        x, y, w, h = boxes[idx]
                        data.append([image_name, FLAGS.size, FLAGS.model, class_name, round(confidences[idx] * 100), x, y, w, h])
                        idx_index += 1
                    else:
                        idxs = np.delete(idxs, idx_index)
                        #idx_index += 1

                frame, x_y_center = BoundingBoxes.draw_bounding_boxes(image, labels, boxes, confidences, classIDs, idxs, colors)
                frame, img_x_y_center = Frame.image_center(frame)

                # Calculo Camera (flir Duo)
                sensor_width = 7.4  # mm
                sensor_height = 5.55  # mm
                focal_lenght = 8  # mm
                camera_height = 10  # m
                height, width = frame.shape[:2]

                GSD = gsd(sensor_width, camera_height, focal_lenght, width)  # metros/pixel
                #print(GSD)
                distances = Frame.distance(x_y_center, img_x_y_center, GSD)

                frame = Frame.draw_dist(frame, x_y_center, img_x_y_center, distances)

                cv2.imwrite(f'{FLAGS.output}/{image_name}_{FLAGS.size}_{FLAGS.model}.jpg', frame)

                i += 1
                print(i, 'of', len(os.listdir(FLAGS.data_path)), 'images')
            else:
                print('Image has ended, failed or wrong path was given.')
                break

        df = pd.DataFrame(data, columns=['image_name', 'net_size', 'model', 'class', 'score', 'x', 'y', 'w', 'h'])
        df['distance'] = distances

        df.to_csv(f"{FLAGS.output}/extracting_bbox_{camera_height}m_{GSD}gsd.csv", index=False)

        cv2.destroyAllWindows()
    else:
        cap = cv2.VideoCapture(FLAGS.data_path)

        video_name = FLAGS.data_path.split("/")[-1]
        video_name = video_name.split(".")[0]

        data = []
        stop = 0
        i = 0
        while True:
            if stop == 0:
                ret, frame = cap.read()
                if ret:
                    # net, layer_names, image, confidence, threshold, net_height, net_width
                    boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, frame,
                                                                                         0.01, 0.03, 960, 960)
                    idx_index = -1
                    for class_id, score, bbox, idx in zip(classIDs, confidences, boxes, idxs):
                        class_name = labels[class_id]
                        if class_name == 'person':
                            idxs = np.delete(idxs, idx_index)
                    idx_index += 1

                    frame = BoundingBoxes.draw_bounding_boxes(frame, labels, boxes, confidences, classIDs, idxs, colors)
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


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
