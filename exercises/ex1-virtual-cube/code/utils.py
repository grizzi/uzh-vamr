import numpy as np

def vector_to_skew(x: np.ndarray) -> np.ndarray:
	return np.array([[0, -x[2], x[1]], 
		             [x[2], 0, -x[0]], 
		             [-x[1], x[0], 0]])

def rodriguez(a: np.ndarray) -> np.ndarray:
	
	angle = np.linalg.norm(a)
	if angle > 0.0:
		vector = a / angle
	else:
		raise NameError("Angle cannot be zero")

	m = np.eye(3, 3)
	k = vector_to_skew(vector)
	m += np.sin(angle) * k + (1 - np.cos(angle)) * k.dot(k)
	return m

def read_poses(file):
	poses = []
	with open(file, 'r') as stream:
		for line in stream:
			row = [float(x) for x in line.split()]
			rotation = np.array(row[:3])
			translation = np.array(row[3:])
			pose = np.zeros(shape=(3, 4))
			pose[:3, :3] = rodriguez(rotation)
			pose[:, 3] = translation
			poses.append(pose)
	return poses

def read_intrinsics(file):
	with open(file, 'r') as stream:
		return np.array([[float(x) for x in line.split()] 
			              for line in stream])

def read_distortion(file):
	with open(file, 'r') as stream:
		params = [[float(x) for x in line.split()] for line in stream]
		return params[0]

