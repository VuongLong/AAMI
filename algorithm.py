import numpy as np
import random
from preprocess import normalize
from utils import *


class Interpolation_T():
	def __init__(self, AN, A, A0, A1, OA1, AN_MeanMat, K):
		self.AN = AN
		self.A = A
		self.A0 = A0
		self.K = K
		self.A1 = A1
		self.OA1 = OA1
		self.AN_MeanMat = AN_MeanMat
		self.missing_map = np.zeros(A1.shape)
		self.missing_map[np.where(A1 == 0)] = 1

		self.UN = mysvd(np.matmul(AN, AN.T))
		self.U0 = [mysvd(np.matmul(A0[i], A0[i].T)) for i in range(self.K)]
		self.U = [mysvd(np.matmul(A[i], A[i].T)) for i in range(self.K)]

		self.U0A0 = [np.matmul(self.U0[i].T, self.A0[i]) for i in range(self.K)]
		self.UA = [np.matmul(self.U[i].T, self.A[i]) for i in range(self.K)]
		self.UNA = [np.matmul(self.UN.T, self.A[i]) for i in range(self.K)]


		self.T = []
		for i in range(self.K):
			U0A0_U0A0T = np.matmul(self.U0A0[i], self.U0A0[i].T)
			UA_U0A0T = np.matmul(self.UA[i], self.U0A0[i].T)
			self.T.append(np.matmul(UA_U0A0T, np.linalg.inv(U0A0_U0A0T)))

	def compute_weighted_mapping(self, weights):
		
		UNA_U0A0T = np.zeros(np.matmul(self.UNA[0], self.U0A0[0].T).shape)
		U0A0_U0A0T = np.zeros(np.matmul(self.U0A0[0], self.U0A0[0].T).shape)
		for i in range(self.K):
			UNA_U0A0T += weights[i] * np.matmul(self.UNA[i], self.U0A0[i].T)
			U0A0_U0A0T += weights[i] * np.matmul(self.U0A0[i], self.U0A0[i].T)

		weighted_T = np.matmul(UNA_U0A0T, np.linalg.inv(U0A0_U0A0T))
		return weighted_T

	def interpolate(self, original_A1, T, mode=-1):
		A1, A1_MeanMat = normalize(original_A1)	
		U1 = mysvd(np.matmul(A1, A1.T))
		if mode == -1:
			A_star = np.matmul(np.matmul(np.matmul(self.UN, T), U1.T), A1)
		else:
			A_star = np.matmul(np.matmul(np.matmul(self.UN, T), U1.T), A1)

		A_star = A_star + A1_MeanMat

		interpolated_A1 = np.copy(original_A1)
		interpolated_A1[np.where(original_A1 == 0)] = A_star[np.where(original_A1 == 0)]
		return interpolated_A1

class Interpolation_F():
	def __init__(self, AN, A, A0, A1, OA1, K):
		self.AN = AN
		self.A = A
		self.A0 = A0
		self.K = K
		self.A1 = A1
		self.OA1 = OA1
		self.missing_map = np.zeros(A1.shape)
		self.missing_map[np.where(A1 == 0)] = 1

		self.VN = mysvd(np.matmul(AN_T, AN))
   		self.V0 = [mysvd(np.matmul(A0[i].T, A0[i])) for i in range(self.K)]

	def compute_weighted_mapping(self, weights):
		A0V0 = [np.matmul(self.A0[i], self.V0[i]) for i in range(self.K)]
		AV = [np.matmul(self.A[i], self.VN) for i in range(self.K)]
		
		A0V0T_A0V0 = np.zeros(np.matmul(A0V0[0].T, A0V0[0]).shape)
		A0V0T_AV = np.zeros(np.matmul(A0V0[0].T, AV[0]).shape)
		for i in range(self.K):
			A0V0T_A0V0 += np.matmul(A0V0[i].T, A0V0[i])
			A0V0T_AV += np.matmul(A0V0[i].T, AV[i])

		weighted_F = np.matmul(np.linalg.inv(A0V0T_A0V0), A0V0T_AV )
		return weighted_F

	def interpolate(self, original_A1, F):
		A1, A1_MeanMat = normalize(original_A1)	
		V1 = mysvd(np.matmul(A1.T, A1))
		A_star = np.matmul(np.matmul(np.matmul(A1, V1), F), V.T)
		A_star = A_star + A1_MeanMat

		interpolated_A1 = np.copy(original_A1)
		interpolated_A1[np.where(original_A1 == 0)] = A_star[np.where(original_A1 == 0)]
		return interpolated_A1





