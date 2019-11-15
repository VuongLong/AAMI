import numpy as np
from utils import MSE


def sign_threshold(error, threshold):
	if error <= threhold:
		return 0
	return 1

def computer_sign_errors(original_A, original_A0, interpolation, mapping, threshold):
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		interpolated_A = interpolation.interpolate(original_A0[i], mapping)
		errors[i] = sign_threshold(MSE(original_A[i], interpolated_A, interpolation.missing_map), threshold)
	return errors

def computer_errors(original_A, original_A0, interpolation, mapping):
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		interpolated_A = interpolation.interpolate(original_A0[i], mapping)
		errors[i] = MSE(original_A[i], interpolated_A, interpolation.missing_map)
	return errors

def ADABOOST(original_A, original_A0, interpolation, R):
	mappings, alphas, final_mapping_errors, final_mappings= [], [], [], []
	weights = np.array([1.0 / interpolation.K for i in range(interpolation.K)])

	for r in range(R):
		# Normalize sample weights
		weights = weights / np.sum(weights)
		#print(weights*100)
		# Compute general mapping matrix
		mapping = interpolation.compute_weighted_mapping(weights*1000)

		# Computer errors and total_error
		errors = computer_errors(original_A, original_A0, interpolation, mapping)
		total_error = np.sum(weights*errors)
		# Compute beta
		beta = total_error / (1 - total_error)

		# Compute alpha
		#alpha = np.log10(1 / beta)

		# Compute new weights
		weights = errors-np.min(errors)

		#store mapping matrix and alpha
		mappings.append(mapping)
		#alphas.append(alpha)
		alphas.append(total_error)

		#
		#
		#
		# Compute strong mapping matrix and observing errors
		weighted_alphas = np.array(alphas, dtype=float)
		weighted_alphas = weighted_alphas / np.sum(weighted_alphas)

		final_mapping = np.zeros(mappings[0].shape)
		for i in range(len(weighted_alphas)):
			final_mapping += weighted_alphas[i] * mappings[i]
		errors = computer_errors(original_A, original_A0, interpolation, final_mapping)
		
		final_mapping_error = np.sum(errors)
		final_mappings.append(final_mapping)
		final_mapping_errors.append(final_mapping_error)
		print('round: ', r, final_mapping_error)
	return final_mappings[-1], final_mapping_errors

