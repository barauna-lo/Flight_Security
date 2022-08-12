import os

import pandas as pd
from absl import app, flags
from absl.flags import FLAGS
import cv2
import numpy as np

from helpers.frame import Frame
from performance.bounding_boxes import BoundingBoxes
from performance.confusion_matriz import ConfusionMatriz
from performance.confusion_matriz_metrics import ConfusionMatrizMetrics
from performance.yolo_predictions import YoloPredictions
from helpers.net_size import change_net_size
from coordinates.geo_coordinates import gsd, geo_to_meter, read_csv_proa_ref
from coordinates.geo_coordinates import read_csv_geo_ref
from coordinates.geo_coordinates import get_coordinates
from helpers.frame import Frame

flags.DEFINE_string('cfg', './detections/cfg/yolov4-tiny_training_640.cfg', 'path to cfg file')
flags.DEFINE_integer('size', 640, 'resize net to')
flags.DEFINE_string('model', 'yolo_tiny_custom_1', 'tiny or yolov4')
flags.DEFINE_string('weights', './detections/weights/yolov4-tiny_training_640_best.weights', 'path to weights file')
flags.DEFINE_string('data_path', './detections/frames/15_undistorted', 'path to frames or video')
flags.DEFINE_string('labeled_path', './detections/contem_pessoas/10m_yolo_annotations', 'path bbox labeled manually')
flags.DEFINE_string('output', './detections/extracted_bbox', 'path to output bboxes')
flags.DEFINE_string('classes', './detections/classes/person.names', 'path to classes name')
flags.DEFINE_string('data_type', 'frame', 'set video or frame')
flags.DEFINE_float('sensor_width', 7.4, 'Camera sensor width')  # mm
flags.DEFINE_float('sensor_height', 5.55, 'Camera sensor_height')  # mm
flags.DEFINE_float('focal_length', 5.374, 'Camera focal_length')  # mm
flags.DEFINE_float('camera_height', 20, 'Camera amera_height')  # mm
flags.DEFINE_boolean('save_data', False, 'save data into csv file')
flags.DEFINE_boolean('save_frames', True, 'save frames')
flags.DEFINE_boolean('confusion_matrix', False, 'evaluating detections results ')
flags.DEFINE_boolean('save_frames_cm', False, 'save frames confusion matrix')

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
        confusion_matriz_result = []
        # loading images
        for frame in os.listdir(FLAGS.data_path):

            # remove .jpg or any image type from image name
            image_name = frame.split(".")[0]

            # case img name has added a new item as in undistorted
            image_name = image_name.split("_")[0:3]
            image_name = "_".join(image_name)

            image = cv2.imread(os.path.join(FLAGS.data_path, frame))
            height, width = image.shape[:2]

            # Find image center point
            img_x_y_center = Frame.image_center(height, width)

            if not image is None:
                print(image_name, ':')
                # net, layer_names, image, confidence, threshold, net_height, net_width
                boxes, confidences, classIDs, idxs = YoloPredictions.make_prediction(net, layer_names, image,
                                                                                     0.005, 0.005, FLAGS.size,
                                                                                     FLAGS.size)
                # class filtering
                idx_index = 0
                for idx in idxs:
                    class_name = labels[classIDs[idx]]
                    if class_name == "person":
                        print(class_name, round(confidences[idx] * 100), "%")
                        idx_index += 1
                    else:
                        idxs = np.delete(idxs, idx_index)
                        # idx_index += 1

                # extract best bbox from class filtering
                bbox_data = BoundingBoxes.bbox_class_filter(labels, boxes, confidences, classIDs, idxs)

                # extract bbox center points
                only_bbox_predicted = []
                for data in bbox_data:
                    only_bbox_predicted.append([data[2], data[3], data[4], data[5]])
                x_y_center = BoundingBoxes.center_bbox(only_bbox_predicted)

                # Calculating GSD
                GSD = gsd(FLAGS.sensor_width, FLAGS.camera_height, FLAGS.focal_length, width)  # metros/pixel
                #print(GSD)

                # Calculating distance
                dist = Frame.distance(x_y_center, img_x_y_center, GSD)

                # lat long
                # ponto de reference 25 no IEAv
                lat_ref = -23.252989
                long_ref = -45.85718722

                ref_x, ref_y = read_csv_geo_ref('/home/ellentuane/Documents/IC/Flight Security/detections/contem_pessoas/geo_reference_15m.csv', image_name)
                proa = read_csv_proa_ref('/home/ellentuane/Documents/IC/Flight Security/detections/contem_pessoas/Dados voo Flir Duo R.csv', image_name)
                ref_point = [int(ref_x), int(ref_y)]
                lat_estimated = []
                long_estimated = []

                # estimating drone position
                drone_x, drone_y = img_x_y_center[0], img_x_y_center[1]
                drone_delta_x = drone_x - int(ref_x)
                drone_delta_y = drone_y - int(ref_y)
                drone_delta_y = - drone_delta_y

                drone_lat_estim, drone_long_estim = get_coordinates(drone_delta_x, drone_delta_y, proa, GSD, lat_ref, long_ref)

                # estimating obj position
                for bb_center in x_y_center:
                    x, y = bb_center[0], bb_center[1]
                    delta_x = x - int(ref_x)
                    delta_y = y - int(ref_y)
                    delta_y = - delta_y

                    lat_estim, long_estim = get_coordinates(delta_x, delta_y, proa, GSD, lat_ref, long_ref)
                    lat_estimated.append(lat_estim)
                    long_estimated.append(long_estim)

                #geo_to_meters = geo_to_meter(lat_estimated, long_estimated, lat_ref, long_ref)
                geo_to_meters = geo_to_meter(lat_estimated, long_estimated, drone_lat_estim, drone_long_estim)

                # gathering all data in one variable
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
                    bbox_data[item].append(lat_estimated[item])
                    bbox_data[item].append(long_estimated[item])
                    bbox_data[item].append(geo_to_meters[item])

                # adding data to data frame variable
                for complete in range(len(bbox_data)):
                    df_data.append(bbox_data[complete])

                if FLAGS.confusion_matrix:
                    # only_bbox_predicted
                    ground_truth = BoundingBoxes.bb_labeled(FLAGS.labeled_path, f'{image_name}.txt', height, width)
                    tp = ConfusionMatriz.true_positive(ground_truth, only_bbox_predicted, 1)
                    fp = ConfusionMatriz.false_positive(tp, only_bbox_predicted)
                    fn = ConfusionMatriz.false_negative(tp, ground_truth)

                    cm = ConfusionMatrizMetrics.confusionMatrixMetrics(len(tp), len(fp), len(fn), len(ground_truth))

                    confusion_matriz_result.append(
                        [image_name, FLAGS.size, len(tp), len(fp), len(fn), cm.precision, cm.recall, cm.accuracy, cm.f1_score,
                         len(only_bbox_predicted), len(ground_truth)])

                    if FLAGS.save_frames_cm:
                        tp1 = []
                        for tps in tp:
                            tp1.append(tps[1])
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(image, ground_truth, (0, 255, 255)) #yellow
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, tp1, (0, 255, 0)) # green
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, fp, (255, 0, 0)) # blue
                        img = BoundingBoxes.draw_bounding_boxes_confusion_matriz(img, fn, (0, 0, 255)) # red

                        cv2.imwrite(f'{FLAGS.output}/{image_name}_{FLAGS.size}_{FLAGS.model}_cm.jpg', img)

                if FLAGS.save_frames:
                    # Draw bboxes in the image
                    frame = BoundingBoxes.draw_bounding_boxes(image, labels, boxes, confidences, classIDs, idxs, colors)
                    frame = BoundingBoxes.draw_center_bbox(frame, only_bbox_predicted)

                    # Draw image center lines
                    #frame = Frame.draw_image_center(frame, height, width)

                    # Draw line from image center to bbox center
                    #frame = Frame.draw_dist(frame, x_y_center, img_x_y_center, dist)

                    # Draw line from ref point to bbox center
                    #frame = Frame.draw_dist(frame, x_y_center, ref_point, geo_to_meters)

                    # Draw line from img center point to bbox center
                    frame = Frame.draw_dist(frame, x_y_center, img_x_y_center, geo_to_meters)

                    # Save final image
                    cv2.imwrite(f'{FLAGS.output}/{image_name}_{FLAGS.size}_{FLAGS.model}.jpg', frame)

                i += 1
                print(i, 'of', len(os.listdir(FLAGS.data_path)), 'images')
            else:
                print('Image has ended, failed or wrong path was given.')
                break

        if FLAGS.save_data:
            # Save data into csv file
            df = pd.DataFrame(df_data,
                              columns=['class', 'score', 'x', 'y', 'w', 'h', 'x_center', 'y_center', 'distance',
                                       'image_name', 'net_size', 'model', 'GSD', 'img_x_center', 'img_y_center',
                                       'camera_height', 'lat_estimaated', 'long_estimated','geo to meters'])
            df.to_csv(f"{FLAGS.output}/extracting_bbox_{FLAGS.camera_height}m_{FLAGS.size}_{FLAGS.model}.csv", index=False)

        if FLAGS.confusion_matrix:
            df = pd.DataFrame(confusion_matriz_result,
                              columns=['frame', 'net_size', 'TP', 'FP', 'FN', 'precision', 'recall',
                                       'accuracy', 'f1_score', 'total_detected', 'total_labeled'])
            df.to_csv(f"{FLAGS.output}/confusion_matriz_{FLAGS.camera_height}m_{FLAGS.size}_{FLAGS.model}.csv", index=False)

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
