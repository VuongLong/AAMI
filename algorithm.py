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



class Interpolation8th_F():
	def __init__(self, reference_matrix, missing_matrix):
		self.A1 = missing_matrix
		self.AN_F = reference_matrix
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries = self.normalization()
		self.K = 0
		self.list_A = []
		self.list_A0 = []
		self.list_V = []
		self.list_V0 = []
		self.list_F = []
		self.fix_leng = missing_matrix.shape[0]

		self.compute_svd()
		self.weight_sample = [1.0/self.K]*self.K

	def normalization(self):
		AA = np.copy(self.combine_matrix)
		weightScale = 200
		MMweight = 0.02
		[frames, columns] = AA.shape
		columnindex = np.where(AA == 0)[1]
		frameindex = np.where(AA == 0)[0]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		framewithgap = np.unique(frameindex)
		Data_without_gap = np.delete(AA, columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		Data = np.copy(AA - MeanMat)
		Data[np.where(AA == 0)] = 0
		
		# calculate weight vector 
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		weight_vector = np.zeros((len(markerwithgap), columns//3))
		for x in range(len(markerwithgap)):
			weight_matrix = np.zeros((frames, columns//3))
			weight_matrix_coe = np.zeros((frames, columns//3))
			for i in range(frames):
				valid = True
				if euclid_dist([0, 0, 0] , get_point(Data, i, markerwithgap[x])) == 0 :
					valid = False
				if valid:
					for j in range(columns//3):
						if j != markerwithgap[x]:
							point1 = get_point(Data, i, markerwithgap[x])
							point2 = get_point(Data, i, j)
							tmp = 0
							if euclid_dist(point2, [0, 0, 0]) != 0:
								weight_matrix[i][j] = euclid_dist(point2, point1)
								weight_matrix_coe[i][j] = 1

			sum_matrix = np.sum(weight_matrix, 0)
			sum_matrix_coe = np.sum(weight_matrix_coe, 0)
			weight_vector_ith = sum_matrix / sum_matrix_coe
			weight_vector_ith[markerwithgap[x]] = 0
			weight_vector[x] = weight_vector_ith
		weight_vector = np.min(weight_vector, 0)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		M_zero = np.copy(Data)
		# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
		N_nogap = np.delete(Data, framewithgap, 0)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0
		
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


		m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
		m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		
		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

		M_zero = np.multiply(((M_zero-m1) / m2),m3)
		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		N_zero = np.multiply(((N_zero-m6) / m5),m33)
		
		m7 = np.ones((Data.shape[0],1))*mean_N_nogap

		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat]

	def compute_svd(self):
		M_zero = self.normed_matries[0]
		N_nogap = self.normed_matries[1]
		N_zero = self.normed_matries[2]
		r = self.fix_leng
		l = 0
		PQ_size = N_zero.shape[1]
		ksmall = 0
		while l <= r:
			self.K += 1
			tmp = np.copy(N_nogap[l:r])
			self.list_A.append(np.copy(tmp))

			tmp = np.copy(N_zero[l:r])
			self.list_A0.append(np.copy(tmp))
			
			_, tmp_V0sigma, tmp_V0 = np.linalg.svd(self.list_A0[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_V0sigma))
			self.list_V0.append(np.copy(tmp_V0.T))
			_, tmp_Vsigma, tmp_V = np.linalg.svd(self.list_A[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_Vsigma))
			self.list_V.append(np.copy(tmp_V.T))
			r += self.fix_leng
			l += self.fix_leng
			r = min(r, N_zero.shape[0])

		for i in range(self.K):
			self.list_V[i] = self.list_V[i][:, :ksmall]
			self.list_V0[i] = self.list_V0[i][:, :ksmall]
			self.list_F.append(np.matmul(self.list_V0[i].T, self.list_V[i]))

	def interpolate_missing(self):
		M_zero = self.normed_matries[0]
		reconstructData = np.copy(M_zero)
		tmp_result = np.zeros(self.A1.shape)
		for i in range(self.K):
			tmp_result += self.weight_sample[i] * np.matmul(np.matmul(np.matmul(M_zero[-self.fix_leng:], self.list_V0[i]), 
											self.list_F[i]), self.list_V[i].T)

		reconstructData[-self.fix_leng:] = tmp_result
		m8 = np.ones((reconstructData.shape[0],1))*self.reconstruct_matries[1]
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), self.reconstruct_matries[2])
		reconstructData = self.reconstruct_matries[0] + (np.multiply(reconstructData, m8) / m3)
		tmp = reconstructData + self.reconstruct_matries[3]
		result = np.copy(tmp)

		final_result = np.copy(self.combine_matrix)
		final_result[np.where(self.combine_matrix == 0)] = result[np.where(self.combine_matrix == 0)]
		return final_result[-self.A1.shape[0]:,:]

	def interpolate_sample(self):
		list_error = []
		for sample_idx in range(self.K):
			current_missing_sample = self.list_A0[sample_idx]
			current_original_sample = self.list_A[sample_idx]
			tmp_result = np.zeros(current_missing_sample.shape)
			for i in range(self.K):
				tmp_result += self.weight_sample[i] * np.matmul(np.matmul(np.matmul(current_missing_sample, 
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



class Interpolation16th_F():
	def __init__(self, reference_matrix, missing_matrix):
		self.A1 = missing_matrix
		self.AN_F = reference_matrix
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries = self.normalization()
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

	def normalization(self):
		AA = np.copy(self.combine_matrix)
		weightScale = 200
		MMweight = 0.02
		[frames, columns] = AA.shape
		columnindex = np.where(AA == 0)[1]
		frameindex = np.where(AA == 0)[0]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		framewithgap = np.unique(frameindex)
		Data_without_gap = np.delete(AA, columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		Data = np.copy(AA - MeanMat)
		Data[np.where(AA == 0)] = 0
		
		# calculate weight vector 
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		weight_vector = np.zeros((len(markerwithgap), columns//3))
		for x in range(len(markerwithgap)):
			weight_matrix = np.zeros((frames, columns//3))
			weight_matrix_coe = np.zeros((frames, columns//3))
			for i in range(frames):
				valid = True
				if euclid_dist([0, 0, 0] , get_point(Data, i, markerwithgap[x])) == 0 :
					valid = False
				if valid:
					for j in range(columns//3):
						if j != markerwithgap[x]:
							point1 = get_point(Data, i, markerwithgap[x])
							point2 = get_point(Data, i, j)
							tmp = 0
							if euclid_dist(point2, [0, 0, 0]) != 0:
								weight_matrix[i][j] = euclid_dist(point2, point1)
								weight_matrix_coe[i][j] = 1

			sum_matrix = np.sum(weight_matrix, 0)
			sum_matrix_coe = np.sum(weight_matrix_coe, 0)
			weight_vector_ith = sum_matrix / sum_matrix_coe
			weight_vector_ith[markerwithgap[x]] = 0
			weight_vector[x] = weight_vector_ith
		weight_vector = np.min(weight_vector, 0)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		M_zero = np.copy(Data)
		# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
		N_nogap = np.delete(Data, framewithgap, 0)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0
		
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


		m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
		m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		
		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

		M_zero = np.multiply(((M_zero-m1) / m2),m3)
		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		N_zero = np.multiply(((N_zero-m6) / m5),m33)
		
		m7 = np.ones((Data.shape[0],1))*mean_N_nogap

		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat]

	def compute_svd(self):
		M_zero = self.normed_matries[0]
		N_nogap = self.normed_matries[1]
		N_zero = self.normed_matries[2]
		r = self.fix_leng
		l = 0
		PQ_size = N_zero.shape[1]
		ksmall = 0
		while l <= r:
			self.K += 1
			tmp = np.copy(N_nogap[l:r])
			self.list_A.append(np.copy(tmp))

			tmp = np.copy(N_zero[l:r])
			self.list_A0.append(np.copy(tmp))
			
			_, tmp_V0sigma, tmp_V0 = np.linalg.svd(self.list_A0[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_V0sigma))
			self.list_V0.append(np.copy(tmp_V0.T))
			_, tmp_Vsigma, tmp_V = np.linalg.svd(self.list_A[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_Vsigma))
			self.list_V.append(np.copy(tmp_V.T))
			r += self.fix_leng
			l += self.fix_leng
			r = min(r, N_zero.shape[0])

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

		self.list_alpha = np.linalg.lstsq(left_hand, right_hand, rcond = None)[0]

		return 0

	def interpolate_missing(self):
		M_zero = self.normed_matries[0]
		reconstructData = np.copy(M_zero)
		tmp_result = np.zeros(self.A1.shape)
		for i in range(self.K):
			tmp_result += diag_matrix(self.list_alpha[i], self.list_F[i].shape[0]) * np.matmul(np.matmul(np.matmul(M_zero[-self.fix_leng:], self.list_V0[i]), 
											self.list_F[i]), self.list_V[i].T)

		reconstructData[-self.fix_leng:] = tmp_result
		m8 = np.ones((reconstructData.shape[0],1))*self.reconstruct_matries[1]
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), self.reconstruct_matries[2])
		reconstructData = self.reconstruct_matries[0] + (np.multiply(reconstructData, m8) / m3)
		tmp = reconstructData + self.reconstruct_matries[3]
		result = np.copy(tmp)

		final_result = np.copy(self.combine_matrix)
		final_result[np.where(self.combine_matrix == 0)] = result[np.where(self.combine_matrix == 0)]
		return final_result[-self.A1.shape[0]:,:]

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