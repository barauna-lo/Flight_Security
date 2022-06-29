import math
import numpy as np
# import cv2
# from utm import from_latlon


def get_dist(x, y, GSD, ref_x, ref_y):
    alvo = np.array((x, y))
    centro_img = np.array((ref_x, ref_y))
    dist = np.linalg.norm(alvo - centro_img)
    return dist*GSD


def get_angle(x, y):
    alvo = np.array([x, y])
    dist = np.linalg.norm(alvo)
    cateto_x = x
    cos_angulo = cateto_x / dist
    # esse valor de angulo estará sempre entre 0 e 90 graus.
    angulo = math.acos(cos_angulo) * (180 / 3.1415)
    if x >= 0 and y >= 0:
        return 90 - angulo
    elif x < 0 and y >= 0:
        return 90 - angulo
    elif x < 0 and y < 0:
        return angulo - 270
    elif x >= 0 and y < 0:
        return angulo + 90


def rotate(x, y, proa):
    az = proa*math.pi/180
    matriz_rotacao = np.array([[math.cos(az), math.sin(az)], [- math.sin(az), math.cos(az)]])
    posit = np.array([x, y])
    return np.matmul(matriz_rotacao, posit)


def get_coordinates(delta_x, delta_y, proa, GSD, lat_ref, long_ref, raio_terra=6.371e6):
    lat_center = lat_ref*(math.pi/180)
    long_center = long_ref*(math.pi/180)
    a, b = rotate(delta_x, delta_y, proa)
    posit = np.array([a*GSD, b*GSD])
    metric_tensor = np.array([[0, 1/raio_terra], [1/(raio_terra*math.cos(lat_center)), 0]])
    delta = np.matmul(metric_tensor, posit)
    new_lat = lat_center + delta[0]
    new_long = long_center + delta[1]
    return new_lat*(180/math.pi), new_long*(180/math.pi)


def gsd(sw, ch, fl, iw):
    # grounding sample distance
    # sw = sensor width
    # fl = focal length
    # ch = camera height
    # iw = image width
    return (sw * ch) / (fl * iw)  # Distance from camera in meters

# Calculo Camera (Phantom 4 PRO)
sensor_width = 13.2     # mm
focal_lenght = 8.8      # mm
altura = 80.02  # m
image_width = 5472
image_height = 3648

# GSD = (sensor_width * altura) / (focal_lenght * image_width)  # metros
GSD = 0.0219    # metros/pixel


# proa = 145.2
#
# # ponto 25
# x = 4529
# y = 246
#
# # ponto referencia 49
# lat_ref = -23.25357772
# long_ref = -45.85711764
# ref_x = 3559
# ref_y = 899
# lat_real = -23.25385303
# long_real = -45.85720856


# proa = 0
#
# # ponto teste
# x = 3559
# y = 890
#
# # ponto referencia teste
# lat_ref = 0
# long_ref = 0
# ref_x = 3559
# ref_y = 889


proa = 145.2

# ponto 11
x = 1098
y = 2565

# ponto referencia 49
lat_ref = -23.25357772
long_ref = -45.85711764
ref_x = 3559
ref_y = 899
lat_real = -23.25298989
long_real = -45.85691000


delta_x = x - ref_x
delta_y = y - ref_y
delta_y = - delta_y

lat_estim, long_estim = get_coordinates(delta_x, delta_y, proa, GSD, lat_ref, long_ref)

#print('esti = (', lat_estim, ',', long_estim, ')')
#print('real = (', lat_real, ',', long_real, ')')