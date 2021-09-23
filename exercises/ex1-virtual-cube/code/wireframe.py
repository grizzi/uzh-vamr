import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

IMAGE_FILE_PATH = "../images/img_0001.jpg"
POSES_FILE_PATH = "../data/poses.txt"
CALIBRATION_FILE_PATH = "../data/K.txt"
DISTORTION_FILE_PATH = "../data/D.txt"

# get checkboard poses from file 
poses = read_poses(POSES_FILE_PATH)

# read intrinsic
K = read_intrinsics(CALIBRATION_FILE_PATH)

# generate checkboard corners 
cell_size = 0.04
x = np.arange(0, 4) * cell_size
y = np.arange(0, 4) * cell_size
xv, yv = np.meshgrid(x, y)

# reshape as a matrix N x 3
N = xv.size()
points = np.zeros(shape=(N, 3))
points[:, 0] = xv.reshape(N, 1)
points[:, 1] = yv.reshape(N, 1)

# transform checkboard points in camera image
hom_coord = K.dot(poses[0]).dot(points)
pixels = hom_coord[:, 2] / hom_coord[:, 3]

# use flag 0 to read as grayscale image
img_gray = cv2.imread(IMAGE_FILE_PATH, 0)

# plots points on the image
fig, ax = plt.subplots()
ax.imshow(img_gray)
ax.scatter(pixels[:, 0], pixels[:, 1], 'o', col="r", markersize=4)
plt.show()
