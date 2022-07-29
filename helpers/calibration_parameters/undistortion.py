import os
import numpy as np
import cv2


def undistorted(frame, matrix_path, dist_path):
    # Load parameters
    matrix = np.load(matrix_path)
    dist = np.load(dist_path)

    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))

    #Undistort images
    frame_undistorted = cv2.undistort(frame, matrix, dist, None, new_camera_matrix)

    ##Uncomment if you want help lines:
    #frame_undistorted = cv2.line(frame_undistorted, (0,int(h/2)), (w,240), (0, 255, 0) , 5)
    #frame_undistorted = cv2.line(frame_undistorted, (int(w/2),0), (int(w/2),hR), (0, 255, 0) , 5)

    return frame_undistorted

matrix_path = '/home/ellentuane/Documents/IC/Flight Security/helpers/calibration_parameters/FLIR_DUO/matrix.npy'
dist_path = '/home/ellentuane/Documents/IC/Flight Security/helpers/calibration_parameters/FLIR_DUO/distortion.npy'
data_path = '/home/ellentuane/Documents/IC/Flight Security/detections/frames'
save_path = '/home/ellentuane/Documents/IC/Flight Security/detections/undistorted'

matrix = np.load(matrix_path)
dist = np.load(dist_path)


for frame in os.listdir(data_path):
    image_name = frame.split(".")[0]
    image = cv2.imread(os.path.join(data_path, frame))

    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, dist, (w, h), 1, (w, h))
    frame_undistorted = cv2.undistort(image, matrix, dist, None, new_camera_matrix)

    cv2.imwrite(f'{save_path}/{image_name}_undistorted.jpg', frame_undistorted)
    print('ok')
