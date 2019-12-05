import numpy as np
from preprocess import generate_patch_data, normalize
from adaboost import *
from algorithm import * 
from utils import *
from data import AN_T, AN_F, A1, original_A1

def train_adaboost(AN, A1, interpolattion_type):
	normalized_AN, AN_MeanMat = normalize(AN)
	original_A, original_A0, K = generate_patch_data(AN, A1, interpolattion_type)
	normalized_A, normalized_A0 = [], []
	for i in range(K):
		normalized_A_i, _ = normalize(original_A[i])
		normalized_A0_i, _ = normalize(original_A0[i])
		normalized_A.append(normalized_A_i)
		normalized_A0.append(normalized_A0_i)

	if interpolattion_type == 'T':
		interpolation = Interpolation_T(normalized_AN, normalized_A, normalized_A0, A1, original_A1, AN_MeanMat, K)
	else:
		interpolation = Interpolation_F(normalized_AN, normalized_A, normalized_A0, A1, original_A1, AN_MeanMat, K)

	mapping, errors = ADABOOST(original_A, original_A0, interpolation, 1000)
	#mapping, errors = Test_ADABOOST(original_A, original_A0, interpolation, 30)
	#mapping, errors = ADABOOST_multi_filter(original_A, original_A0, interpolation, 10)
	return mapping, errors

if __name__ == '__main__':

	print("Reference source:")
	print('AN_T', AN_T.shape)
	print('AN_F', AN_F.shape)

	print("\nTest source:")
	print('A1', A1.shape)
	
	print("\nCompute Mapping Matrix")
	mapping, errors = train_adaboost(AN_T, A1, 'T')