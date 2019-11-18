import numpy as np
import random

'''
consider AN in T and F
	T is normal
	F we should convert to T case to get AN mean
'''
def normalize(A):
	A = A.T
	A_MeanVec = A.sum(0) / (A != 0).sum(0)
	A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
	normalized_A = np.copy(A - A_MeanMat)
	normalized_A[np.where(A == 0)] = 0 # value can be different from zeros after minusing mean
	return np.copy(normalized_A.T), np.copy(A_MeanMat.T)

def generate_patch_data_T(AN, A1):
	K = AN.shape[1] // A1.shape[1] 
	patch_length = A1.shape[1] 
	A, A0= [], []
	for i in range(K):
		tmp = np.copy(AN[:, i * patch_length:(i+1)  * patch_length])
		A.append(np.copy(tmp))

		tmp[np.where(A1 == 0)] = 0
		A0.append(np.copy(tmp))

	return A, A0, K

def generate_patch_data_F(AN, A1):
	K = AN.shape[0] // A1.shape[0] 
	patch_length = A1.shape[0] 
	A, A0= [], []
	for i in range(K):
		tmp = np.copy(AN[i * patch_length:(i+1)  * patch_length, :])
		A.append(np.copy(tmp))

		tmp[np.where(A1 == 0)] = 0
		A0.append(np.copy(tmp))

	return A, A0, K

def generate_patch_data(AN, A1, interpolattion_type):
	if interpolattion_type == 'T':
		return generate_patch_data_T(AN, A1)
	return generate_patch_data_F(AN, A1)

def generate_missing_joint(n, m, frame_length, number_gap):
	matrix = np.ones((n,m))
	counter = 0
	joint_in = []
	while counter < number_gap:
		counter+=1
		tmp = arg.cheat_array[counter-1]
		# tmp = np.random.randint(0, m//3)
		# while tmp in joint_in:
		# 	tmp = np.random.randint(0, m//3)
		# 	joint_in.append(tmp)
			
		start_missing_frame = np.random.randint(0, n - frame_length)
		missing_joint = tmp
		# print("start_missing_frame: ", start_missing_frame, "joint: ", missing_joint)
		for frame in range(start_missing_frame, start_missing_frame+frame_length):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
	counter = 0
	for x in range(n):
		for y in range(m):
			if matrix[x][y] == 0: counter +=1
	print("percent missing: ", 100 * counter / (n*m))
	return matrix