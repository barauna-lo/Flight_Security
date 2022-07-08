import os
import cv2
import numpy as np


class Frame:
    def __init__(self):
        pass

    @staticmethod
    def create_dir(input_path, output_path):
        name = input_path.split("/")[-1].split(".")[0]
        save_path = os.path.join(output_path, name)
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        except OSError:
            print(f"ERROR: creating directory with name {save_path}")
        return save_path, name

    @staticmethod
    def save_frame(save_path, name, gap, frame, idx):
        if idx == 0:
            cv2.imwrite(f"{save_path}/{name}_{idx}_.jpg", frame)
        else:
            if idx % gap == 0:
                cv2.imwrite(f"{save_path}/{name}_{idx}_.jpg", frame)

    @staticmethod
    def video_frame(video_path, save_dir, gap):
        cap = cv2.VideoCapture(video_path)
        idx = 0
        save_path, name = Frame.create_dir(video_path, save_dir)
        while True:
            ret, frame = cap.read()
            if ret:
                Frame.save_frame(save_path, name, gap, frame, idx)
            else:
                cap.release()
                break
            idx += 1

    @staticmethod
    def rescaleFrame(frame, scale=0.75):
        # images, video and live videos
        height = int(frame.shape[0] * scale)
        width = int(frame.shape[1] * scale)
        dimensions = (width, height)

        return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

    @staticmethod
    def image_center(height, width):
        return [int(width / 2), int(height / 2)]

    @staticmethod
    def draw_image_center(image, height, width):
        start_point_horizontal = (0, int(height / 2))
        end_point_horizontal = (width, int(height / 2))

        cv2.line(image, start_point_horizontal, end_point_horizontal, (255, 0, 0), 1)

        start_point_vertical = (int(width / 2), 0)
        end_point_vertical = (int(width / 2), height)

        cv2.line(image, start_point_vertical, end_point_vertical, (255, 0, 0), 1)
        return image

    @staticmethod
    def draw_dist(frame, bbox_x_y_center, img_x_y_center, distance):
        i = 0
        for bb_center in bbox_x_y_center:
            cv2.line(frame, (img_x_y_center[0], img_x_y_center[1]), (bb_center[0], bb_center[1]), (255, 0, 0), 2)
            cv2.putText(frame, f'{distance[i]} m', (bb_center[0] + 50, bb_center[1]), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 255), 2)
            i += 1
        return frame

    @staticmethod
    def distance(bbox_x_y_center, img_x_y_center, GSD):
        distances = []
        img_x_y_center = np.array(img_x_y_center)

        for bb_center in bbox_x_y_center:
            bb_center = np.array((bb_center[0], bb_center[1]))
            dist_pix = np.linalg.norm(bb_center - img_x_y_center)  # valor em Pixel
            dist = round(dist_pix * GSD, 2)
            distances.append(dist)
        return distances

    @staticmethod
    def click_event(event, x, y, flags, params):
        # function to display the pixel coordinates
        # of the points clicked on the image  # ix, iy = [], []

        global ix, iy
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            ix.append(x), iy.append(y)
            print('dentro da funcao', ix, iy)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image', frame)
