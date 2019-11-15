import numpy as np
from preprocess import generate_patch_data, normalize
from adaboost import ADABOOST
from algorithm import Interpolation_T, Interpolation_F 
from utils import *
from data import AN_T, AN_F, A1, original_A1

def train_adaboost(AN, A1, interpolattion_type):
	normalized_AN, _ = normalize(AN)
	normalized_A1, _ = normalize(A1)
	
	original_A, original_A0, K = generate_patch_data(AN, A1, interpolattion_type)
	normalized_A, normalized_A0, K = generate_patch_data(normalized_AN, normalized_A1, interpolattion_type)

	if interpolattion_type == 'T':
		interpolation = Interpolation_T(normalized_AN, normalized_A, normalized_A0, A1, K)
	else:
		interpolation = Interpolation_F(normalized_AN, normalized_A, normalized_A0, A1, K)

	mapping, errors = ADABOOST(original_A, original_A0, interpolation, 1000)
	return mapping, errors

if __name__ == '__main__':

	print("Reference source:")
	print('AN_T', AN_T.shape)
	print('AN_F', AN_F.shape)

	print("\nTest source:")
	print('A1', A1.shape)
	
	print("\nCompute Mapping Matrix")
	mapping, errors = train_adaboost(AN_T, A1, 'T')