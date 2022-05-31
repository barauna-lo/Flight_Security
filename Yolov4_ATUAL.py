import math
import numpy as np
from time import time
from tensorflow import lite, constant, shape, reshape
from tensorflow.image import combined_non_max_suppression
from tensorflow.saved_model import load as load_saved_model
from tensorflow.config.experimental import list_physical_devices, set_memory_growth
from tensorflow.python.saved_model import tag_constants
from tensorflow.compat.v1 import ConfigProto
from absl.app import run
from absl.flags import FLAGS, DEFINE_string, DEFINE_integer, DEFINE_boolean, DEFINE_float
from cv2 import VideoCapture, imwrite, INTER_AREA, FONT_HERSHEY_SIMPLEX, LINE_4, VideoWriter_fourcc, VideoWriter, putText, line, rectangle, \
    imshow, waitKey, destroyAllWindows, cvtColor, resize, COLOR_BGR2RGB, COLOR_RGB2BGR, CAP_PROP_FRAME_WIDTH, \
    CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS
from numpy import array, asarray, newaxis, float32, delete, linspace
from matplotlib.pyplot import get_cmap
from shapely.geometry import LineString, Point
from deep_sort.preprocessing import non_max_suppression
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools.generate_detections import create_box_encoder
from YOLOtoTF.yolov4 import filter_boxes
from YOLOtoTF.config import cfg
from YOLOtoTF.utils import format_boxes, read_class_names
import matplotlib
from pymongo import MongoClient
import datetime

from dronekit import connect, VehicleMode

try:
    set_memory_growth(list_physical_devices('GPU')[0], True)
except:
    pass

DEFINE_string("framework", "tf", "(tf, tflite, trt)")
DEFINE_string("weights", "./Networks/TensorFlow/Model", "Path to weights file")
#DEFINE_string("weights", "/home/piter/PycharmProjects/DeepsortProject/model_coco", "Path to weights file")
DEFINE_string("model", "yolov4", "(yolov3,  yolov4)")
#DEFINE_string("video", "Esquerda.mp4", "Path to input video or set to 0 for webcam")
DEFINE_string("video", "Meio.mp4", "Path to input video or set to 0 for webcam")
DEFINE_string("output", "output.mp4", "Path to output video")
DEFINE_string("output_format", "XVID", "Codec used in VideoWriter when saving video to file")
DEFINE_float("iou", 0.45, "IOU threshold")
DEFINE_float("score", 0.50, "Score threshold")
DEFINE_integer("size", 416, "Resize images to...")
DEFINE_boolean("tiny", False, "If present, the YOLO-tiny presets are used instead of the standard full YOLO")
DEFINE_boolean("dont_show", False, "If present, don't show video output")
DEFINE_boolean("info", True, "If present, show detailed info of tracked objects")
DEFINE_boolean("count", True, "If present, count objects being tracked on screen")

# connection_string = '/dev/ttyACM0'
#
# print('Connecting to vehicle on: %s' % connection_string)
# vehicle = connect(connection_string, wait_ready=True)

# cluster = "mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&directConnection=true&ssl=false"
# client = MongoClient(cluster)
#
# db = client.Test


def main(_argv):
    # Definition of some hyperparameters:
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_overlap = 1.0

    # Initialize DeepSORT:
    model_filename = 'model_data/mars-small128.pb'
    encoder = create_box_encoder(model_filename, batch_size=1)

    # Calculate the cosine distance metric:
    metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)

    # Initialize tracker
    tracker = Tracker(metric)

    # Load configuration for the object detector:
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    input_size = FLAGS.size
    video_path = FLAGS.video

    # Load TensorFlow saved model:
    if (FLAGS.framework != "tflite"):
        saved_model_loaded = load_saved_model(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    # Otherwise, load TFlite saved model:
    else:
        interpreter = lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

    # Start video's pointer, using webcam if the video_path is 0:
    try:
        vid = VideoCapture(int(video_path))
    except:
        vid = VideoCapture(video_path)
        if vid.isOpened():
            width = vid.get(3)
            height = vid.get(4)
            print(width)
            print(height)

    # If is set to have an output, it starts its pointer too as well:
    out = None
    if (FLAGS.output):
        out = VideoWriter(FLAGS.output, VideoWriter_fourcc(*FLAGS.output_format), int(vid.get(CAP_PROP_FPS)),
                          (int(vid.get(CAP_PROP_FRAME_WIDTH)), int(vid.get(CAP_PROP_FRAME_HEIGHT))))

    if (FLAGS.info is True):
        timeseries = open("timeseries.csv", 'w')
        timeseries.write("Frame,ID,Class,X,Y,Xmin,Ymin,Xmax,Ymax,Color\n")
        timeseries.close()

    frame_num = 0

    # The structure bellow is referent to the frames sequence:
    while (1 < 2):
        return_value, frame = vid.read()
        if (return_value is True):
            frame = cvtColor(frame, COLOR_BGR2RGB)
        else:
            print("There's nothing more to process here.")
            break

        frame_num += 1
        print('Frame #: ', frame_num)

        image_data = (resize(frame, (input_size, input_size)) / 255)[newaxis, ...].astype(float32)
        start_time = time()
        # print(image_data.shape)

        # Run detections (if is using a standard TensorFlow framework):
        if (FLAGS.framework != "tflite"):
            for key, value in infer(constant(image_data)).items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # This block is for TFlite frameworks):
        else:
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

            # This block is for YOLOv3 networks only:
            if ((FLAGS.model == "yolov3") and (FLAGS.tiny is True)):
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                input_shape=constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                input_shape=constant([input_size, input_size]))

        boxes, scores, classes, valid_detections = combined_non_max_suppression(
            boxes=reshape(boxes, (shape(boxes)[0], -1, 1, 4)),
            scores=reshape(pred_conf, (shape(pred_conf)[0], -1, shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score)

        original_h, original_w, _ = frame.shape

        # Convert the data to numpy arrays and slice out of them unused elements and
        # format the bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
        num_objects = valid_detections.numpy()[0]
        scores = scores.numpy()[0][0:int(num_objects)]
        classes = classes.numpy()[0][0:int(num_objects)]
        bboxes = format_boxes(boxes.numpy()[0][0:int(num_objects)], original_h, original_w)

        # Store all predictions in one parameter for simplicity when calling functions
        # pred_bbox = [bboxes, scores, classes, num_objects]

        # Get the classes' names:
        class_names = read_class_names(cfg.YOLO.CLASSES)

        # It gonna use by default all the classes in .names file...
        allowed_classes = list(class_names.values())
        # allowed_classes = ['Car']  # ... but here you can also set a subset of classes if you like to, just uncomment here.

        # Loop through objects and use class index to get class name (only classes in allowed_classes list):
        names = []
        deleted_indx = []
        for i in range(num_objects):
            class_name = class_names[int(classes[i])]
            if (class_name not in allowed_classes):
                deleted_indx.append(i)
            else:
                names.append(class_name)

        if (FLAGS.count is True):
            count = len(names)
            putText(frame, "Objects being tracked: {}".format(count), (5, 35), 0, 1, (0, 255, 0), 2)
            print("Objects being tracked: {}".format(count))

        # Delete detections that are not in allowed_classes:
        if (len(deleted_indx) > 0):
            bboxes = delete(bboxes, deleted_indx, axis=0)
            scores = delete(scores, deleted_indx, axis=0)

        # Initialize the color map:
        colors = [get_cmap("tab20b")(i)[:3] for i in linspace(0, 1, 20)]

        # Encode yolo detections and feed the tracker:
        detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                      zip(bboxes, scores, array(names), encoder(frame, bboxes))]

        # Call the tracker with non-maxima supression:
        tracker.predict()
        tracker.update([detections[i] for i in non_max_suppression(array([d.tlwh for d in detections]),
                                                                   array([d.class_name for d in detections]),
                                                                   nms_max_overlap,
                                                                   array([d.confidence for d in detections]))])

        # Update the tracks:
        for track in tracker.tracks:
            if ((not track.is_confirmed()) or (track.time_since_update > 1)):
                continue
            bbox = track.to_tlbr()
            class_name = track.get_class()
            id = track.track_id
            centroid = (((bbox[0] + bbox[2]) / 2), ((bbox[1] + bbox[3]) / 2))
            x = ((bbox[0] + bbox[2]) / 2)
            y = ((bbox[1] + bbox[3]) / 2)
            print(int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3]))
            # print("centroid:",centroid)
            # print(x)
            # print(y)
            v = (frame.shape[1] / 2)
            w = (frame.shape[0] / 2)
            print(v)
            print(w)
            # Distancia Euclideana

            an = np.array((x, y))
            bn = np.array((v, w))
            dist = np.linalg.norm(an - bn)  # valor em Pixel
            an2 = array((x, 540))
            bn2 = np.array((v, 540))
            cateto = np.linalg.norm(an2 - bn2)

            # Calculo Camera (Phantom 4 PRO)
            Sensor_width = 13.2  # mm
            Focal_Lenght = 8.8  # mm
            Altura = 50  # m
            Image_Width = width
            Image_Height = height
            GSD = (((Sensor_width * Altura) * 100) / (Focal_Lenght * Image_Width)) / 100  # metros
            # print("GSD:", GSD)
            dist = dist * GSD
            cateto = cateto * GSD
            print("dist (m)=", dist)
            print("cateto (m):", cateto)
            Cos_angulo = cateto / dist
            # print("Cos_Angulo:",Cos_angulo)
            start_point = (int(x), int(y))
            end_point = (int(v), int(w))
            # YAW PIXHAWK
            # yaw_angle = vehicle.attitude.yaw
            # yaw_angle = (yaw_angle * (180 / 3.1415)) + 180
            # print("yaw:", yaw_angle)
            Cos_angulo = math.acos(Cos_angulo) * (180 / 3.1415)
            print("Cosseno do angulo (Graus)=", Cos_angulo)
            print("distancia:", Point(x, y).distance(Point(v, w)))  # valor em Pixel
            #linha de calculo
            # line(frame, start_point, end_point, (255, 0, 0), 2)
            # line(frame, (int(x), 540), (int(v), 540), (0, 255, 0), 2)
             # Draw the bounding box and its label on the screen
            color = [i * 255 for i in (colors[int(track.track_id) % len(colors)])]
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            rectangle(frame, (int(bbox[0]), int(bbox[1] - 30)),
                      (int(bbox[0]) + ((len(class_name) + len(str(track.track_id))) * 17), int(bbox[1])), color, -1)
            putText(frame, (class_name + str(id)), (int(bbox[0]), int(bbox[1] - 10)), 0, 0.75, (255, 255, 255), 2)
            if frame_num%10 == 0:
                #print("SHAPE1:", np.array(result[int(ymin): int(ymax), int(xmin):int(xmax)]).shape)
                if (0 in np.array(result[int(ymin): int(ymax), int(xmin):int(xmax)]).shape) == False:
                    imwrite("Imagens_Meio/" +class_name + str(id) + ".png",np.array(result[int(ymin): int(ymax), int(xmin):int(xmax)]))


            # # Quadrante 2
            # if (x < (width/2) and y < (height/2)):
            #     Cos_angulo = Cos_angulo*(-1) +180
            #     print("Cosseno do angulo (Graus)2=", Cos_angulo)
            #     line(frame, (960, 540), (960, 0), (255, 0, 255), 2)
            #     line(frame, (960, 540), (0, 540), (255, 0, 255), 2)
            #     putText(frame, (str(dist) + 'metros'),
            #             (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2)), 0, 0.75,
            #             (0, 0, 255), 2)
            #     putText(frame, (str(Cos_angulo) + ' Graus'), (
            #     int(((start_point[0] + end_point[0]) / 2) + 30), int(((start_point[1] + end_point[1]) / 2) + 30)), 0,
            #             0.75, (255, 0, 0), 2)
            # # Quadrante 3
            # elif (x < (width/2) and y > (height/2) and y < height):
            #     Cos_angulo = Cos_angulo + 180
            #     print("Cosseno do angulo (Graus)3=", Cos_angulo)
            #     line(frame, (0, 540), (960, 540), (255, 0, 255), 2)
            #     line(frame, (960, 540), (960, 1080), (255, 0, 255), 2)
            #     putText(frame, (str(dist) + 'metros'),
            #             (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2)), 0, 0.75,
            #             (0, 0, 255), 2)
            #     putText(frame, (str(Cos_angulo) + ' Graus'), (
            #     int(((start_point[0] + end_point[0]) / 2) + 30), int(((start_point[1] + end_point[1]) / 2) + 30)), 0,
            #             0.75, (255, 0, 0), 2)
            # # Quadrante 1
            # elif (x < width and x > (width/2) and y > 0 and y < (height/2)):
            #     Cos_angulo = Cos_angulo + 0
            #     print("Cosseno do Angulo (Graus) 4=", Cos_angulo)
            #     line(frame, (960, 0), (960, 540), (255, 0, 255), 2)
            #     line(frame, (960, 540), (1920, 540), (255, 0, 255), 2)
            #     putText(frame, (str(dist) + 'metros'),
            #             (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2)), 0, 0.75,
            #             (0, 0, 255), 2)
            #     putText(frame, (str(Cos_angulo) + ' Graus'), (
            #     int(((start_point[0] + end_point[0]) / 2) + 30), int(((start_point[1] + end_point[1]) / 2) + 30)), 0,
            #             0.75, (255, 0, 0), 2)
            #
            # # Quadrante 4
            # elif (x > (width / 2) and y > (height / 2)):
            #     Cos_angulo = Cos_angulo * (-1) + 270
            #     print("Cosseno do angulo (Graus)2=", Cos_angulo)
            #     line(frame, (int(width / 2), int(height / 2)), (int(width), int(height / 2)), (255, 0, 255), 2)
            #     line(frame, (int(width / 2), int(height / 2)), (int(width / 2), int(height)), (255, 0, 255), 2)
            #     putText(frame, (str(dist) + 'metros'),
            #             (int((start_point[0] + end_point[0]) / 2), int((start_point[1] + end_point[1]) / 2)), 0, 0.75,
            #             (0, 0, 255), 2)
            #     putText(frame, (str(Cos_angulo) + ' Graus'), (
            #         int(((start_point[0] + end_point[0]) / 2) + 30), int(((start_point[1] + end_point[1]) / 2) + 30)),
            #             0,
            #             0.75, (255, 0, 0), 2)
            # Print info dump if its flag is enable to, with details about each track, and create timeseries from it:
            if (FLAGS.info is True):
                print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id),
                                                                                                    class_name, (
                                                                                                    int(bbox[0]),
                                                                                                    int(bbox[1]),
                                                                                                    int(bbox[2]),
                                                                                                    int(bbox[3]))))
                timeseries = open("timeseries.csv", 'a')
                timeseries.write(str(frame_num) + ','
                                 + str(track.track_id) + ','
                                 + class_name + ','
                                 + str(int((bbox[0] + bbox[2]) / 2)) + ','
                                 + str(int((bbox[1] + bbox[3]) / 2)) + ','
                                 + str(int(bbox[0])) + ','
                                 + str(int(bbox[1])) + ','
                                 + str(int(bbox[2])) + ','
                                 + str(int(bbox[3])) + ','
                                 + str(color) + ','
                                  + '\n')

                # x e y (centroid)
                xmin = str(int(bbox[0]))
                ymin = str(int(bbox[1]))
                xmax = str(int(bbox[2]))
                ymax = str(int(bbox[3]))

                todo1 = {"Frame": frame_num, "Tracker ID": track.track_id, "Class": class_name,
                         "BBOX Coord": [xmin, xmax, ymin, ymax], "BBOX Centroid": [x, y]}
                # todos = db.todos
                # result = todos.insert_one(todo1)

                #todos.update_one({"Frame"})
                # print(result)


                # if frame_num % 50 == 0:
                #     todos.delete_many({})
                # else:
                #     print("Nao faz nada")

                timeseries.close()

        # Measures the FPS rate of the running detections and dump it:

        # v1=((frame.shape[1]/2)-5)
        # print(v1)
        # w1=((frame.shape[0]/2)-5)
        # print(w1)
        # v2=((frame.shape[1] / 2) + 5)
        # print(v2)
        # w2=((frame.shape[0] / 2) + 5)
        # print(w2)
        # rectangle(frame, (int((frame.shape[1] / 2) - 5), int((frame.shape[0] / 2) - 5)),
        #           (int((frame.shape[1] / 2) + 5), int((frame.shape[0] / 2) + 5)), (255, 0, 0), -1)
        font = FONT_HERSHEY_SIMPLEX
        # Lat = '-23.251641'
        # Long = '-45.856884'
        # putText(frame,
        #         ('Lat Long:' + str(Lat) + str(Long)),
        #         (950, 600),
        #         font, 1,
        #         (255, 0, 0),
        #         2,
        #         LINE_4)
        print("FPS: %.2f" % (1.0 / (time() - start_time)))
        result = cvtColor(asarray(frame), COLOR_RGB2BGR)

        if (not FLAGS.dont_show):
            resized = resize(result, (int(width / 2), int(height / 2)), interpolation=INTER_AREA)
            imshow("Output Video", resized)

        # If an output is set, save video file
        if (FLAGS.output):
            out.write(result)

        # Allow you to break the loop if you press Q:
        if (waitKey(1) & 0xFF == ord('q')):
            break

    # Release input and output pointers:
    vid.release()
    out.release()
    destroyAllWindows()


if __name__ == '__main__':
    try:
        run(main)
    except SystemExit:
        pass
