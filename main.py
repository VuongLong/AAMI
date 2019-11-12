import numpy as np
from preprocess import generate_patch_data, normalize
from adaboost import ADABOOST
from algorithm import Interpolation_T, Interpolation_F 

def train_adaboost(AN, A1, interpolattion_type):
	normalized_AN, _ = normalize(AN)
	normalized_A1, _ = normalize(A1)
	
	original_A, original_A0, K = generate_patch_data(AN, A1, interpolattion_type)
	normalized_A, normalized_A0, K = generate_patch_data(normalized_AN, normalized_A1, interpolattion_type)
	if interpolattion_type == 'T':
		interpolation = Interpolation_T(normalized_AN, normalized_A, normalized_A0, K)
	else:
		interpolation = Interpolation_F(normalized_AN, normalized_A, normalized_A0, K)

	mapping, errors = ADABOOST(original_A, original_A0, interpolation, 10)
	return mapping, errors




mapping, errors = train_adaboost((AN, A1, 'T')