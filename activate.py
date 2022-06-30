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
flags.DEFINE_string('data_path', './detections/frames', 'path to frames or video')
flags.DEFINE_string('output', './detections/extracted_bbox', 'path to output bboxes')
flags.DEFINE_string('classes', './detections/classes/coco.names', 'path to classes name video')
flags.DEFINE_string('data_type', 'frame', 'set video or frame')
flags.DEFINE_float('sensor_width', 7.4, 'Camera sensor width') # mm
flags.DEFINE_float('sensor_height', 5.55, 'Camera sensor_height') # mm
flags.DEFINE_float('focal_length', 8, 'Camera focal_length') # mm
flags.DEFINE_float('camera_height', 10, 'Camera amera_height') # mm
flags.DEFINE_boolean('save_data', True, 'save data into csv file')
flags.DEFINE_boolean('save_frames', False, 'save frames')


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
        df_data = []
        # loading images
        for frame in os.listdir(FLAGS.data_path):

            # remove .jpg or any image type from image name
            image_name = frame.split(".")[0]

            image = cv2.imread(os.path.join(FLAGS.data_path, frame))
            height, width = image.shape[:2]

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
                        idx_index += 1
                    else:
                        idxs = np.delete(idxs, idx_index)
                        #idx_index += 1

                # Draw bboxes in the image
                frame, bbox_data = BoundingBoxes.draw_bounding_boxes(image, labels, boxes, confidences, classIDs, idxs, colors)
                # draw bbox center points
                boxes = []
                for data in bbox_data:
                    boxes.append([data[2], data[3], data[4], data[5]])
                frame, x_y_center = BoundingBoxes.center_bbox(frame, boxes)
                # Find image center point
                frame, img_x_y_center = Frame.image_center(frame)
                # Calculating GSD
                GSD = gsd(FLAGS.sensor_width, FLAGS.camera_height, FLAGS.focal_length, width)  # metros/pixel
                # Calculating distance
                dist = Frame.distance(x_y_center, img_x_y_center, GSD)
                # Draw line from image center to bbox center
                frame = Frame.draw_dist(frame, x_y_center, img_x_y_center, dist)

                for item in range(len(bbox_data)):
                    bbox_data[item].append(x_y_center[item][0])
                    bbox_data[item].append(x_y_center[item][1])
                    bbox_data[item].append(dist[item])
                    bbox_data[item].append(image_name)
                    bbox_data[item].append(FLAGS.size)
                    bbox_data[item].append(FLAGS.model)
                    bbox_data[item].append(GSD)
                    bbox_data[item].append(img_x_y_center[0])
                    bbox_data[item].append(img_x_y_center[1])
                    bbox_data[item].append(FLAGS.camera_height)

                for complete in range(len(bbox_data)):
                    df_data.append(bbox_data[complete])

                if FLAGS.save_frames:
                    # Save final image
                    cv2.imwrite(f'{FLAGS.output}/{image_name}_{FLAGS.size}_{FLAGS.model}.jpg', frame)

                i += 1
                print(i, 'of', len(os.listdir(FLAGS.data_path)), 'images')
            else:
                print('Image has ended, failed or wrong path was given.')
                break

        if FLAGS.save_data:
            # Save data into csv file
            df = pd.DataFrame(df_data, columns=['class', 'score', 'x', 'y', 'w', 'h', 'x_center', 'y_center', 'distance', 'image_name', 'net_size', 'model', 'GSD', 'img_x_center', 'img_y_center', 'camera_height'])
            df.to_csv(f"{FLAGS.output}/extracting_bbox_{FLAGS.camera_height}m_ {FLAGS.size}.csv", index=False)


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
