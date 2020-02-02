import numpy as np
import random
from preprocess import normalize
from utils import *

# this method regards to original method of Dr. Yu 
# normaliztion is diffrent with these previous methods
class Interpolation16_original():
	def __init__(self, reference_matrix, missing_matrix):
		self.A1 = missing_matrix
		self.AN_F = reference_matrix
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normalization()
		self.K = 0
		self.list_A = []
		self.list_A0 = []
		self.list_V = []
		self.list_V0 = []
		self.list_F = []
		self.fix_leng = missing_matrix.shape[0]
		self.list_alpha = []
		self.compute_svd()
		self.weight_sample = [1.0/self.K]*self.K
		self.inner_compute_alpha()


	# in this function, comprising training data and testing data into one package
	# training data and testing data share same meaning matrix
	def normalization(self):
		A = np.copy(self.AN_F)
		A1 = np.copy(self.A1)

		AA1 = self.combine_matrix

		AA1_MeanVec = AA1.sum(0) / (AA1 != 0).sum(0)
		A_MeanMat = np.tile(AA1_MeanVec, (A.shape[0], 1))
		A_new = np.copy(A - A_MeanMat)

		A1_MeanMat = np.tile(AA1_MeanVec,(A1.shape[0], 1))
		A1_new = np.copy(A1 - A1_MeanMat)
		A1_new[np.where(A1 == 0)] = 0
		self.AN_norm = A_new
		self.A1_norm = A1_new
		self.A1_mean = A1_MeanMat
		return np.copy(A_new.T), np.copy(A1_new.T), np.copy(A1_MeanMat.T)

	# in this function, computing meaning matrixes of training data and testing data seperatly
	def normalization_2(self):
		A = np.copy(self.reference_matrix)
		A1 = np.copy(self.A1)

		A_MeanVec = np.mean(A, 0)
		A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
		A_new = np.copy(A - A_MeanMat)

		A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
		A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
		A1_new = np.copy(A1 - A1_MeanMat)
		A1_new[np.where(A1 == 0)] = 0
		self.AN_F_norm = A_new
		self.A1_norm = A1_new
		self.A1_mean = A1_MeanMat
		return np.copy(A_new.T), np.copy(A1_new.T), np.copy(A1_MeanMat.T)

	def compute_svd(self):
		columnindex = np.where(self.A1_norm == 0)[1]
		columnwithgap = np.unique(columnindex)
		r = self.AN_norm.shape[0]
		l = r - self.fix_leng
		ksmall = 0
		while l >= 0:
			self.K += 1
			tmp = np.copy(self.AN_norm[l:r])
			self.list_A.append(np.copy(tmp))

			tmp = np.copy(self.AN_norm[l:r])
			tmp[np.where(self.A1 == 0)] = 0
			
			self.list_A0.append(np.copy(tmp))
			
			_, tmp_V0sigma, tmp_V0 = np.linalg.svd(self.list_A0[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_V0sigma))
			self.list_V0.append(np.copy(tmp_V0.T))
			_, tmp_Vsigma, tmp_V = np.linalg.svd(self.list_A[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_Vsigma))
			self.list_V.append(np.copy(tmp_V.T))
			r -= self.fix_leng
			l -= self.fix_leng

		for i in range(self.K):
			self.list_V[i] = self.list_V[i][:, :ksmall]
			self.list_V0[i] = self.list_V0[i][:, :ksmall]
			self.list_F.append(np.matmul(self.list_V0[i].T, self.list_V[i]))


	def inner_compute_alpha(self):
		# build list_alpha
		list_Q = []
		for iloop in range(self.K):
			for kloop in range(self.K):
				qi = matmul_list([self.list_V[kloop], self.list_F[kloop].T, self.list_V0[iloop].T, self.list_A0[iloop].T, 
						diag_matrix(self.weight_sample[iloop], self.list_A[iloop].shape[0]), self.list_A[iloop]])
				list_Q.append(qi)
		tmp_matrix = np.zeros(list_Q[-1].shape)
		for x in list_Q:
			tmp_matrix += x
		tmp_matrix = tmp_matrix.reshape(tmp_matrix.shape[0] * tmp_matrix.shape[1], 1)
		right_hand = np.copy(tmp_matrix)

		list_P = []
		for jloop in range(self.K):
			list_Pijk = []
			for iloop in range(self.K):
				for kloop in range(self.K):
					Pijk = matmul_list([self.list_V[kloop], self.list_F[kloop].T, self.list_V0[iloop].T, self.list_A0[iloop].T, 
							diag_matrix(self.weight_sample[iloop], self.list_A0[iloop].shape[0]), self.list_A0[iloop], 
							self.list_V0[iloop], self.list_F[jloop], self.list_V[jloop].T ])
					list_Pijk.append(Pijk)
			tmp_matrix = np.zeros(list_Pijk[-1].shape)
			for x in list_Pijk:
				tmp_matrix += x
			shape_x, shape_y = tmp_matrix.shape[0], tmp_matrix.shape[1]
			tmp_matrix = tmp_matrix.reshape(shape_x * shape_y, 1)
			list_P.append(np.copy(tmp_matrix))

		left_hand = np.hstack([ x for x in list_P])

		self.list_alpha = np.linalg.lstsq(np.matmul(left_hand.T, left_hand), np.matmul(left_hand.T, right_hand), rcond = None)[0]

		return 0

	def interpolate_missing_plain(self):
		VN, _, _ = np.linalg.svd(np.matmul(self.AN_norm.T, self.AN_norm)) 
		V0, _, _ = np.linalg.svd(np.matmul(self.A1_norm.T , self.A1_norm)) 

		F = np.copy(np.matmul(V0.T, VN))
		A_star = np.matmul(np.matmul(np.matmul(self.A1_norm, V0), F), VN.T)
		A_result = np.copy(A_star + self.A1_mean)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = A_result[np.where(self.A1 == 0)]
		return final_result

	def interpolate_missing(self):
		tmp_result = np.zeros(self.A1.shape)
		for i in range(self.K):
			tmp_result += diag_matrix(self.list_alpha[i], self.list_F[i].shape[0]) * np.matmul(np.matmul(np.matmul(self.A1_norm, self.list_V0[i]), 
											self.list_F[i]), self.list_V[i].T)
		result = np.copy(self.A1_norm)
		result[np.where(self.A1_norm == 0)] = tmp_result[np.where(self.A1_norm == 0)]
		reconstructed =np.copy(result + self.A1_mean)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = reconstructed[np.where(self.A1 == 0)]
		return final_result

	def interpolate_sample(self):
		list_error = []
		for sample_idx in range(self.K):
			current_missing_sample = self.list_A0[sample_idx]
			current_original_sample = self.list_A[sample_idx]
			tmp_result = np.zeros(current_missing_sample.shape)
			for i in range(self.K):
				tmp_result += diag_matrix(self.list_alpha[i], self.list_F[i].shape[0]) * np.matmul(np.matmul(np.matmul(current_missing_sample, 
												self.list_V0[i]), self.list_F[i]), self.list_V[i].T)
			result_sample = np.copy(current_missing_sample)
			result_sample[np.where(current_missing_sample == 0)] = tmp_result[np.where(current_missing_sample == 0)]
			list_error.append(ARE(result_sample, current_original_sample))
		return list_error


	def get_weight(self):
		return self.weight_sample

	def set_weight(self, weight):
		# weight must be a new object
		for x in range(self.K):
			self.weight_sample[x] = weight[x]

	def get_number_sample(self):
		return self.K

	def get_alpha(self):
		return self.list_alpha
