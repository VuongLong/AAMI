import numpy as np
import random
from preprocess import normalize


class Interpolation_T():
	def __init__(self, AN, A, A0, K):
		self.AN = AN
		self.A = A
		self.A0 = A0
		self.K = K

		self.UN = mysvd(np.matmul(AN, AN.T))
		self.U0 = [mysvd(np.matmul(A0[i], A0[i].T)) for i in range(K)]

	def compute_weighted_mapping(self, weights):
		UNA = [np.matmul(self.UN.T, self.A[i]) for i in range(K)]
		U0A0 = [np.matmul(self.U0[i].T, self.A0[i]) for i in range(K)]

		UNA_U0A0T = np.zeros(np.matmul(UNA[0], U0A0[0].T).shape)
		U0A0_U0A0T = np.zeros(np.matmul(U0A0[0], U0A0[0].T).shape)
		for i in range(K):
			UNA_U0A0T += weights[i] * np.matmul(UNA[i], U0A0[i].T)
			U0A0_U0A0T += weights[i] * np.matmul(U0A0[i], U0A0[i].T)

		weighted_T = np.matmul(UNA_U0A0T, np.linalg.inv(U0A0_U0A0T))
		return weighted_T

	def interpolate(self, original_A1, T):
		A1, A1_MeanMat = normalize(original_A1)	
		U1 = mysvd(np.matmul(A1, A1.T))
		A_star = np.matmul(np.matmul(np.matmul(self.UN, T), U1.T), A1)
		A_star = A_star + A1_MeanMat

		interpolated_A1 = np.copy(original_A1.T)
		interpolated_A1[np.where(original_A1.T == 0)] = A_star[np.where(original_A1.T == 0)]
		return interpolated_A1.T

class Interpolation_F():
	def __init__(self, AN, A, A0, K):
		self.AN = AN
		self.A = A
		self.A0 = A0
		self.K = K

		self.VN = mysvd(np.matmul(AN_T, AN))
   		self.V0 = [mysvd(np.matmul(A0[i].T, A0[i])) for i in range(K)]

	def compute_weighted_mapping(self, weights):
		A0V0 = [np.matmul(self.A0[i], self.V0[i]) for i in range(K)]
		AV = [np.matmul(self.A[i], self.VN) for i in range(K)]
		
		A0V0T_A0V0 = np.zeros(np.matmul(A0V0[0].T, A0V0[0]).shape)
		A0V0T_AV = np.zeros(np.matmul(A0V0[0].T, AV[0]).shape)
		for i in range(K):
			A0V0T_A0V0 += np.matmul(A0V0[i].T, A0V0[i])
			A0V0T_AV += np.matmul(A0V0[i].T, AV[i])

		weighted_F = np.matmul(np.linalg.inv(A0V0T_A0V0), A0V0T_AV )
		return weighted_F

	def interpolate(self, original_A1, F):
		A1, A1_MeanMat = normalize(original_A1)	
		V1 = mysvd(np.matmul(A1.T, A1))
		A_star = np.matmul(np.matmul(np.matmul(A1, V1), F), V.T)
		A_star = A_star + A1_MeanMat

		interpolated_A1 = np.copy(original_A1.T)
		interpolated_A1[np.where(original_A1.T == 0)] = A_star[np.where(original_A1.T == 0)]
		return interpolated_A1.T





