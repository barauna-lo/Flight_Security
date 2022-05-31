import math
import numpy as np
import cv2
from utm import from_latlon


def get_dist(x, y, GSD, ref_x, ref_y):
    alvo = np.array((x, y))
    centro_img = np.array((ref_x, ref_y))
    dist = np.linalg.norm(alvo - centro_img)
    return dist*GSD


def get_angle(x, y):
    alvo = np.array([x, y])
    dist = np.linalg.norm(alvo)
    cateto_x = x    # - ref_x
    cos_angulo = cateto_x / dist
    # print(x)
    # print('cos_angulo =', cos_angulo)
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


def get_azimuth(x, y, proa):
    angulo = get_angle(x, y)
    azimuth = proa + angulo
    if azimuth < 0:
        return (azimuth + 360)*math.pi/180
    return azimuth*math.pi/180


def rotate(x, y, proa):
    az = get_azimuth(x, y, proa)
    print('az =', az)
    matriz_rotacao = np.array([[math.cos(az), math.sin(az)], [math.cos(az), - math.sin(az)]])
    posit = np.array([x, y])
    return np.matmul(matriz_rotacao, posit)


def get_coordinates(x, y, proa, GSD, lat_ref, long_ref, raio_terra=6.371e6):
    lat_center = lat_ref*(math.pi/180)
    long_center = long_ref*(math.pi/180)
    a, b = rotate(x, y, proa)
    posit = np.array([a*GSD, b*GSD])
    metric_tensor = np.array([[1/raio_terra, 0], [0, 1/(raio_terra*math.cos(lat_center))]])
    delta = np.matmul(metric_tensor, posit)
    new_lat = lat_center + delta[0]
    new_long = long_center + delta[1]
    return new_lat*(180/math.pi), new_long*(180/math.pi)


# Calculo Camera (Phantom 4 PRO)
sensor_width = 13.2     # mm
focal_lenght = 8.8      # mm
altura = 80.02  # m
image_width = 5472
image_height = 3648

# GSD = (sensor_width * altura) / (focal_lenght * image_width)  # metros
GSD = 0.0219    # metros/pixel
proa = 145.2

# ponto 49
x = 3559
y = 889

# # ponto 25
# x = 4529
# y = 246

# # ponto teste
# x = 1100
# y = 2566

# ponto referencia 11
lat_ref = -23.25298989
long_ref = -45.85691000
ref_x = 1098
ref_y = 2565


x = x - ref_x
y = y - ref_y
y = 3648 - y
lat_estim, long_estim = get_coordinates(x, y, proa, GSD, lat_ref, long_ref)
# lat_estima, long_estima = get_coordinates(0, 2000, 0, GSD, lat_ref, long_ref)

print('(', lat_estim, ',', long_estim, ')')
