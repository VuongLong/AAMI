import numpy as np

def MSE(A, B):
	return 0

def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	return U