import numpy as np
import random

'''
consider AN in T and F
	T is normal
	F we should convert to T case to get AN mean
'''
def normalize(A):
		A_MeanVec = np.mean(A, 0)
		A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
		normalized_A = np.copy(A - A_MeanMat)
		normalized_A[np.where(A == 0)] = 0 # value can be different from zeros after minusing mean
		return np.copy(normalized_A.T), np.copy(A_MeanMat.T)

def generate_patch_data((AN, A1, interpolattion_type):
	if interpolattion_type == 'T':
		return generate_patch_data_T((AN, A1)
	else:
		return generate_patch_data_F((AN, A1)

def generate_patch_data_T(AN, A1):
	K = AN.shape[1] // A1.shape[1] 
	patch_length = A1.shape[1] 
	A, A0= [], [], []
	for i in range(K):
		tmp = np.copy(AN[:, i * patch_length,(i+1)  * patch_length])
		A.append(np.copy(tmp))

		tmp[np.where(A_1 == 0)] = 0
		A0.append(np.copy(tmp))

	return A, A0, K

def generate_patch_data_F(AN, A1):
	K = AN.shape[0] // A1.shape[0] 
	patch_length = A1.shape[0] 
	A, A0= [], [], []
	for i in range(K):
		tmp = np.copy(AN[i * patch_length,(i+1)  * patch_length, :])
		A.append(np.copy(tmp))

		tmp[np.where(A_1 == 0)] = 0
		A0.append(np.copy(tmp))

	return A, A0, K

	