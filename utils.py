import numpy as np

def diag_matrix(value, k):
	value_array = [value]*k
	return np.diag(value_array)

def matmul_list(matrix_list):
	number_matrix = len(matrix_list)
	result = np.copy(matrix_list[0])
	for i in range(1, number_matrix):
		result = np.matmul(result, matrix_list[i])
	return result


def MSE(A, B, missing_map):
	return np.sum(np.abs(A - B) * (1-missing_map)) / np.sum(missing_map)

def ARE(predict, original):
	return np.mean(np.abs((predict - original) / original))

def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	return U

def remove_joint(data):
	list_del = []
	list_del_joint = [5, 9, 14, 18]

	for x in list_del_joint:
		list_del.append(x*3)
		list_del.append(x*3+1)
		list_del.append(x*3+2)
	data = np.delete(data, list_del, 1)
	#print('removed joints data', data.shape)
	return data 

def euclid_dist(X, Y):
	XX = np.asarray(X)
	YY = np.asarray(Y)
	return np.sqrt(np.sum(np.square(XX - YY)))

def get_zero(matrix):
	counter = 0
	for x in matrix:
		if x > 0.01: counter += 1
	return counter

def get_point(Data, frame, joint):
	point = [ Data[frame, joint*3] , Data[frame, joint*3+1] , Data[frame, joint*3+2 ]]
	return point


def read_tracking_data3D(data_dir, patch):
	print("reading source: ", data_dir, " patch: ", patch)

	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line.split(',')
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	#print('original data', Tracking3D.shape)

	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D[patch[0]: patch[1]]
	#print('patch data', Tracking3D.shape)

	Tracking3D = remove_joint(Tracking3D)
	restore = np.copy(Tracking3D)
	return Tracking3D, restore

def read_tracking_data3D_without_RJ(data_dir, patch):
	#print("reading source: ", data_dir, " patch: ", patch)

	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line.split(' ')
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	#print('original data', Tracking3D.shape)

	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D[patch[0]: patch[1]]
	#print('patch data', Tracking3D.shape)
	
	restore = np.copy(Tracking3D)
	return Tracking3D, restore
