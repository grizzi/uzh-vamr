import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import *

DEBUG = False
CUBE_SIZE = 0.08
IMAGE_FILE_PATH = "../data/images_undistorted/img_0001.jpg"
POSES_FILE_PATH = "../data/poses.txt"
CALIBRATION_FILE_PATH = "../data/K.txt"
DISTORTION_FILE_PATH = "../data/D.txt"

def get_box_coordinates(origin=np.array([0.0, 0.0, 0.0]), size=1.0):
	coords = np.zeros(shape=(4, 8))
	coords[0, :] = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
	coords[1, :] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
	coords[2, :] = [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0]
	coords = coords * size
	
	coords[3, :] = [1.0  for _ in range(8)]

	coords[0, :] += origin[0]
	coords[1, :] += origin[1]
	coords[2, :] += origin[2]
 
	return coords

def draw_cube(coords, ax):
	pair = [(0, 1), (1, 2), (2, 3), (3, 0), 
	        (4, 5), (5, 6), (6, 7), (7, 4), 
	        (0, 4), (1, 5), (2, 6), (3, 7)]

	for v, w in pair:
		ax.plot([coords[0, v], coords[0, w]], 
			    [coords[1, v], coords[1, w]], lw=3, c='r')

def project_points(K, T, coords):
	return K.dot(T).dot(coords)

# get checkboard poses from file 
poses = read_poses(POSES_FILE_PATH)

# read intrinsic
K = read_intrinsics(CALIBRATION_FILE_PATH)
print(f"Camera intrinsics:\nK={K}")

# read distortion parameters
D = read_distortion(DISTORTION_FILE_PATH)
print(f"Camera distortion parameters:\nD={D}")

# generate checkboard corners 
cell_size = 0.04
num_cells = 4
x = np.arange(0, num_cells) * cell_size
y = np.arange(0, num_cells) * cell_size
xv, yv = np.meshgrid(x, y)

# reshape as a matrix N x 3
N = xv.size
points = np.zeros(shape=(4, N))
points[0, :] = xv.reshape(N)
points[1, :] = yv.reshape(N)
points[3, :] = np.ones(shape=(N))

# transform checkboard points in camera image
hom_coord = project_points(K, poses[0], points)
pixels = hom_coord[:2, :] / hom_coord[2, :]

# transform cube to image
hom_coord = project_points(K, poses[0], get_box_coordinates(size=CUBE_SIZE))
pixels_cube = hom_coord[:2, :] / hom_coord[2, :]

# use flag 0 to read as grayscale image
img_gray = cv2.imread(IMAGE_FILE_PATH, 0)


# plots points and cube on the image
fig, ax = plt.subplots()
ax.imshow(img_gray, cmap='gray')

if DEBUG: 
	ax.scatter(pixels[0, :], pixels[1, :], 30, c='b')

# Plot cube 
draw_cube(pixels_cube, ax)
plt.show()
