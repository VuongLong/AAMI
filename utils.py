import numpy as np

def MSE(A, B, missing_map):
	return np.sum(np.abs(A - B) * missing_map) / np.sum(missing_map)

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
