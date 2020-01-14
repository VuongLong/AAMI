import numpy as np
from preprocess import generate_patch_data, normalize
from adaboost import *
from algorithm import * 
from utils import *
from data import AN_T, AN_F, A1, original_A1, missing_map
import copy

class adaboost_16th():
	def __init__(self, inner_function, number_loop = 20):
		self.iteration_lim = number_loop
		self.function = inner_function
		self.list_function = []
		self.list_function.append(self.function)
		self.list_beta = []
		self.threshold = 0.5
		self.power_coefficient = 3
		self.number_sample = inner_function.get_number_sample()
		self.limit_error = 2
		self.list_mean_error = []
		self.partly_accumulate = []

	def set_iteration(self, number):
		self.iteration_lim = number

	def train(self):
		index_maxError = -1
		index_maxPosition = -1
		for loop_i in range(self.iteration_lim):
			print("looping: ", loop_i)
			current_function = self.list_function[-1]
			accumulate_error_weight = 0
			# compute error for each sample
			error_sample = current_function.interpolate_sample()
			self.list_mean_error.append(np.mean(error_sample))
			for x in range(len(error_sample)):
				if error_sample[x] > error_sample[index_maxError]:
					index_maxError = x
				if (error_sample[index_maxError] > self.limit_error) and (index_maxPosition == -1):
					index_maxPosition = loop_i
			
			# self.threshold = np.median(error_sample)
			print("error_sample: ", error_sample)
			print("threshold: ", self.threshold)
			alpha = current_function.get_alpha()
			weight_sample = current_function.get_weight()
			if (loop_i - index_maxPosition >= 10) and (index_maxPosition != -1):
				weight_sample[index_maxError] = 0
				index_maxPosition = -1
			print("weight_sample: ", weight_sample)
			# compute error rate for function
			for x in range(self.number_sample):
				if error_sample[x] > self.threshold:
					accumulate_error_weight += weight_sample[x]
			current_beta = accumulate_error_weight ** self.power_coefficient
			self.list_beta.append(current_beta)
			new_distribution = []
			accumulate_Z = 0
			for x in range(self.number_sample):
				if error_sample[x] <= self.threshold:
					new_distribution.append(weight_sample[x] * current_beta)
				else:
					new_distribution.append(weight_sample[x])
				accumulate_Z += new_distribution[-1]
			print("accumulate_Z: ", accumulate_Z)
			print("accumulate_error_weight: ", accumulate_error_weight)
			if accumulate_error_weight <= 0.00001:
				self.iteration_lim = loop_i-1
				print("/////////////////stop training ADABOOST/////////////")
				break
			if accumulate_error_weight >= 0.9999:
				self.iteration_lim = loop_i
				print("/////////////////stop training ADABOOST/////////////")
				break
			for x in range(self.number_sample):
				new_distribution[x] = new_distribution[x] / accumulate_Z
			print("new_distribution: ", new_distribution)
			self.partly_accumulate.append(self.interpolate_partly_accumulate(loop_i+1))
			if self.partly_accumulate[-1] > last_weight:
				self.iteration_lim = loop_i - 1
				break
			print("finish loop: ", loop_i)
			# update new function
			new_function = copy.deepcopy(current_function)
			new_function.set_weight(np.copy(new_distribution))
			new_function.inner_compute_alpha()
			self.list_function.append(new_function)

	def get_beta_info(self):
		return self.list_beta

	def interpolate_accumulate(self):
		if self.iteration_lim < 1:
			return self.function.interpolate_missing()
		for t in range(self.iteration_lim):
			current_function = self.list_function[t]
			result = current_function.interpolate_missing()
			print("result: ",t," ", MSE(result, original_A1, missing_map))
		list_result = []
		start_round = 0
		if self.iteration_lim >= 2:
			for x in range(min(self.iteration_lim-1, 2)):
				if (self.list_mean_error[x] < np.mean(np.asarray(self.list_mean_error[x+1:self.iteration_lim]))):
					break
				start_round = x+1
		# if (self.list_mean_error[0] > np.mean(np.asarray(self.list_mean_error[1:-1]))) and self.iteration_lim >= 2:
		# 	start_round = 1
		# 	print("///////////////////////////////////////////")
		# 	print("bi tru")
		# 	print("///////////////////////////////////////////")
		for t in range(start_round, self.iteration_lim):
			current_function = self.list_function[t]
			function_result = current_function.interpolate_missing()
			list_result.append(function_result)
		result = np.zeros(list_result[-1].shape)
		sum_beta = 0
		counter = 0
		for t in range(start_round, self.iteration_lim):
			function_beta = np.log(1/self.list_beta[t])
			result += function_beta * list_result[counter]
			counter+= 1 
			sum_beta += function_beta
		final_result = result / sum_beta
		return final_result

	def interpolate_partly_accumulate(self, loop_number):
		list_result = []
		sum_beta = 0
		for t in range(loop_number):
			current_function = self.list_function[t]
			list_error = current_function.interpolate_sample()
			function_beta = np.log(1/self.list_beta[t])
			list_result.append(function_beta * np.asarray(list_error))
			sum_beta += function_beta
		current_result  = list_result[0]
		for t in range(1, loop_number):
			current_result += list_result[t]

		final_result = current_result / sum_beta
		return np.mean(final_result)

	def get_distribution_sample(self):
		list_distri = []
		for function in self.list_function:
			list_distri.append(function.get_weight())
		return	list_distri


	def get_arbitrary_sample(self, sample_idx = -1):
		return self.list_function[sample_idx]

def calculate_mae_matrix(X):
	error_sum = np.sum(np.abs(X))
	mse = np.sum(np.square(X))
	print("debug")
	print("mse: ",mse)
	print("mae: ",error_sum)
	print("end")
	return error_sum / len(X)

if __name__ == '__main__':

	print("Reference source:")
	print('AN_T', AN_T.shape)
	print('AN_F', AN_F.shape)

	print("\nTest source:")
	print('A1', A1.shape)
	
	interpolation = Interpolation16th_F(AN_F, A1)
	result1 = interpolation.interpolate_missing()
	result2 = interpolation.debug
	print("result1: ", MSE(result1, original_A1, missing_map))
	print("result2: ", MSE(result2, original_A1, missing_map))
	boosting = adaboost_16th(interpolation) 
	boosting.train()
	# print(boosting.get_beta_info())
	result2 = boosting.interpolate_accumulate()
	print(np.around(calculate_mae_matrix(original_A1[np.where(missing_map == 0)]- result2[np.where(missing_map == 0)]), decimals = 17))

