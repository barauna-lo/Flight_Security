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

# from coordinates.pixels_coordinate import click_event
from coordinates.geo_coordinates import gsd

flags.DEFINE_string('cfg', './detections/cfg/yolov4.cfg', 'path to cfg file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('model', 'tiny', 'tiny or yolov4')
flags.DEFINE_string('weights', './detections/weights/yolov4.weights', 'path to weights file')
flags.DEFINE_string('data_path', './detections/frames', 'path to frames or video')
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
        # loading images
        for frame in os.listdir(FLAGS.data_path):
            # to every image in the folder, a .txt file will be created
            #data = []

            # remove .jpg or any image type from image name
            image_name = frame.split(".")[0]
            image = cv2.imread(os.path.join(FLAGS.data_path, frame))

            if not image is None:

                # net, layer_names, image, confidence, threshold, net_height, net_width
                boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, image,
                                                                                     0.01, 0.03, FLAGS.size, FLAGS.size)

                print(image_name, ':')
                idx_index = 0
                for class_id, score, bbox, idx in zip(classIDs, confidences, boxes, idxs):

                    class_name = labels[class_id]
                    if class_name == 'person':
                        print(class_name, int(score * 100), "%")
                        x, y, w, h = bbox
                        data.append([image_name, FLAGS.size, FLAGS.model, class_name, round(score * 100), x, y, w, h])
                    else:
                        idxs = np.delete(idxs, idx_index)

                frame = BoundingBoxes.draw_bounding_boxes(image, labels, boxes, confidences, classIDs, idxs, colors)
                frame = Frame.image_center(frame)

                # Calculo Camera (Phantom 4 PRO)
                sensor_width = 7.4  # mm
                sensor_height = 5.55  # mm
                focal_lenght = 8  # mm
                camera_height = 10  # m
                height, width = frame.shape[:2]

                gsd = gsd(sensor_width, camera_height, focal_lenght, width)
                print(gsd)

                proa = 145.2  # graus

                #ponto referencia 11
                lat_ref = -23.25298989
                long_ref = -45.85691000
                ref_x = 911
                ref_y = 579

                cv2.imwrite(f'{FLAGS.output}/{image_name}_{FLAGS.size}_{FLAGS.model}.jpg', frame)


                i += 1
                print(i, 'of', len(os.listdir(FLAGS.data_path)), 'images')
            else:
                print('Image has ended, failed or wrong path was given.')
                break

        df = pd.DataFrame(data, columns=['image_name', 'net_size', 'model', 'class', 'score', 'x', 'y', 'w', 'h'])
        df.to_csv(f"{FLAGS.output}/extracting_bbox.csv", index=False)

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
