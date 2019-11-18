import numpy as np
from utils import MSE


def sign_threshold(error, threshold):
	if error <= threshold:
		return 0
	return 1

def make_one_sample_error(original_A, original_A0, interpolation, mapping):
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		interpolated_A = interpolation.interpolate(original_A0[i], mapping)
		errors[i] = MSE(original_A[i], interpolated_A, interpolation.missing_map)
	errors[np.where(errors==np.max(errors))] = 0
	errors[np.where(errors!=0)] = 1	
	return 1 - errors

def compute_sign_errors(original_A, original_A0, interpolation, mapping, threshold):
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		interpolated_A = interpolation.interpolate(original_A0[i], mapping, i)
		errors[i] = sign_threshold(MSE(original_A[i], interpolated_A, interpolation.missing_map), threshold)
	return errors

def compute_errors(original_A, original_A0, interpolation, mapping, mode=-1):
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		if mode != -1:
			mode = i
		interpolated_A = interpolation.interpolate(original_A0[i], mapping, mode)
		errors[i] = MSE(original_A[i], interpolated_A, interpolation.missing_map)
	return errors

def ADABOOST(original_A, original_A0, interpolation, R):
	mappings, alphas, final_mapping_errors, final_mappings= [], [], [], []
	weights = np.array([1.0 for i in range(interpolation.K)])

	for r in range(R):
		# Compute general mapping matrix
		mapping = interpolation.compute_weighted_mapping(weights)

		# Computer errors and total_error
		errors = make_one_sample_error(original_A, original_A0, interpolation, mapping)
		
		total_error = np.sum(weights*errors)
		# Compute beta
		beta = min(total_error / (1 - total_error),0.9)
		# Compute alpha
		alpha = np.log10(1 / beta)

		# Compute new weights
		for i in range(interpolation.K):
			weights[i] = weights[i] * (beta ** (1-errors[i]))
		#store mapping matrix and alpha
		mappings.append(mapping)
		#alphas.append(alpha)
		alphas.append(alpha)

		#
		#
		#
		# Compute strong mapping matrix and observing errors
		weighted_alphas = np.array(alphas, dtype=float)
		weighted_alphas = weighted_alphas / np.sum(weighted_alphas)

		final_mapping = np.zeros(mappings[0].shape)
		for i in range(len(weighted_alphas)):
			final_mapping += alphas[i] * mappings[i]
		errors = compute_errors(original_A, original_A0, interpolation, final_mapping)
		
		final_mapping_error = np.sum(errors)
		final_mappings.append(final_mapping)
		final_mapping_errors.append(final_mapping_error)
		print('round: ', r, final_mapping_error)
		interpolated_A1 = interpolation.interpolate(interpolation.A1, mapping)
		errors_A1 = MSE(interpolation.OA1, interpolated_A1, interpolation.missing_map)
		print('A1', errors_A1)
	return final_mappings[-1], final_mapping_errors

def Test_ADABOOST(original_A, original_A0, interpolation, R):
	mappings, alphas, final_mapping_errors, final_mappings= [], [], [], []

	weights = np.array([1.0 for i in range(interpolation.K)])
	mapping = interpolation.compute_weighted_mapping(weights)

		# Computer errors and total_error
	errors_all= compute_errors(original_A, original_A0, interpolation, mapping)
	for r in range(interpolation.K):
		# Normalize sample weights
		weights = np.array([1.0 for i in range(interpolation.K)])
		weights[r] = 100.0
		#print(weights)
		# Compute general mapping matrix
		mapping = interpolation.compute_weighted_mapping(weights)

		# Computer errors and total_error
		errors = compute_errors(original_A, original_A0, interpolation, mapping)
		print(errors[r] - errors_all[r])
	return 0, 0

def calculate_threshold(original_A, original_A0, interpolation):

	errors = np.zeros(interpolation.K * interpolation.K)
	for k in range(interpolation.K):
		mapping = interpolation.T[k]
		errors[k*interpolation.K:(k+1)*interpolation.K] = compute_errors(original_A, original_A0, interpolation, mapping, k)
	threshold = max(errors)
	for v in range(interpolation.K*interpolation.K):
		count = 0
		for i in range(interpolation.K):
			per = 0.0
			for j in range(interpolation.K):
				if errors[i*interpolation.K + j] > errors[v]:
					per += 1.0	
			if per / interpolation.K <= 0.05:
				count += 1
		if count == interpolation.K and errors[v] < threshold:
			threshold = errors[v]

	print(threshold, errors)
	if threshold == max(errors):
		threshold = -1
	return threshold

def ADABOOST_multi_filter(original_A, original_A0, interpolation, R):
	mappings, alphas, final_mapping_errors, final_mappings= [], [], [], []
	weights = np.array([1.0 / interpolation.K for i in range(interpolation.K)])
	threshold = calculate_threshold(original_A, original_A0, interpolation)
	if threshold == -1:
		print('can not find a threshold')
		return 0, 0
	print('threshold', threshold)
	
	for r in range(R):
		best_mapping = 0
		best_total_error = 1

		# Normalize sample weights
		weights = weights / np.sum(weights)
		for k in range(interpolation.K):
			
			# Compute general mapping matrix
			mapping = interpolation.T[k]

			# Computer errors and total_error
			errors = compute_sign_errors(original_A, original_A0, interpolation, mapping, threshold)
			print(errors)
			total_error = np.sum(weights*errors)
			
			if total_error < best_total_error:
				best_total_error = total_error
				best_errors = errors
				best_mapping = k

		print(best_mapping)
		# Compute beta
		beta = best_total_error / (1 - best_total_error)
		# Compute alpha
		alpha = np.log10(1 / beta)

		# Compute new weights
		#weights = best_errors - np.min(best_errors)
		# Compute new weights
		for i in range(interpolation.K):
			#print('b', weights[i])
			weights[i] = weights[i] * (beta ** (1-best_errors[i]))
			#print('a', beta, (1-errors[i]), beta ** (1-errors[i]), weights[i])
		
		#store mapping matrix and alpha
		mappings.append(interpolation.TN[best_mapping])
		#alphas.append(alpha)
		alphas.append(alpha)
		print('a', alpha)
		#
		#
		#
		# Compute strong mapping matrix and observing errors
		weighted_alphas = np.array(alphas, dtype=float)
		weighted_alphas = weighted_alphas / np.sum(weighted_alphas)

		final_mapping = np.zeros(mappings[0].shape)
		for i in range(len(weighted_alphas)):
			final_mapping += alphas[i] * mappings[i]
		errors = compute_errors(original_A, original_A0, interpolation, final_mapping)
		
		final_mapping_error = np.sum(errors)
		final_mappings.append(final_mapping)
		final_mapping_errors.append(final_mapping_error)
		print('round: ', r, final_mapping_error)
		interpolated_A1 = interpolation.interpolate(interpolation.A1, mapping)
		errors_A1 = MSE(interpolation.OA1, interpolated_A1, interpolation.missing_map)
		print('A1', errors_A1)
	return final_mappings[-1], final_mapping_errors

