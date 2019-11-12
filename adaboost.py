import numpy as np
from utils import MSE


def sign_threhold(error, threhold):
	if error <= threhold:
		return 0
	else:
		return 1

def computer_sign_errors(original_A, original_A0, interpolation, mapping, threhold)
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		interpolated_A = interpolation.interpolate(original_A0, mapping)
		errors[i] = sign_threhold(MSE(original_A, interpolated_A), threhold)
	return errors

def computer_errors(original_A, original_A0, interpolation, mapping)
	errors = np.zeros(interpolation.K)
	for i in range(interpolation.K):
		interpolated_A = interpolation.interpolate(original_A0, mapping)
		errors[i] = MSE(original_A, interpolated_A)
	total_error = np.sum(weights*errors)
	return errors, total_error

def ADABOOST(original_A, original_A0, interpolation, R):
	mappings, alphas, final_mapping_errors, final_mappings= [], [], [], []
	weights = np.array([1.0 / interpolation.K for i in range(interpolation.K)])

	for r in range(R):
		# Normalize sample weights
		weights = weights / np.sum(weights)

		# Compute general mapping matrix
		mapping = interpolation.compute_weighted_T(weights)

		# Computer errors and total_error
		errors = computer_errors(original_A, original_A0, interpolation, mapping)
		total_error = np.sum(weights*errors)

		# Compute beta
		beta = total_error / (1 - total_error)

		# Compute alpha
		alpha = np.log10(1 / beta)

		# Compute new weights
		weights = errors

		#store mapping matrix and alpha
		mappings.append(mapping)
		alphas.append(alpha)
	
		# Compute strong mapping matrix and observing errors
		final_mapping = np.zeros(mappings[0].shape)
		for i in range(R)
			final_mapping += alphas[i] * mappings[i]
		errors = computer_errors(original_A, original_A0, interpolation, final_mapping)
		final_mapping_error = np.sum(weights*errors)
		final_mappings.append(final_mapping)
		final_mapping_errors.append(final_mapping_error)

	return final_mappings[-1], final_mapping_errors
