from utils import *

print('READING REFERENCE DATA')

REFERENCE_DIR = [
"./data/135_02.txt",
"./data/135_02.txt"]
SELECTED_DATA = [
[50, 450], 
[850, 2050]]

PATCH_LENGTH = 400
list_patch = []
for data_iter in range(len(REFERENCE_DIR)):
	reference_data, _  = read_tracking_data3D(REFERENCE_DIR[data_iter], SELECTED_DATA[data_iter], False)

	K = reference_data.shape[0] // PATCH_LENGTH
	
	list_patch += [np.copy(reference_data[i*PATCH_LENGTH:(i+1)*PATCH_LENGTH]) for i in range(K)]
	print(len(list_patch))
AN_T = np.hstack(list_patch)
AN_F = np.vstack(list_patch)



#print('\n\n\nREADING TEST DATA')
TEST_DIR = ["./data/135_02.txt"]
SELECTED_TEST_DATA = [[450, 850]]
data_iter = 0
test_data, _  = read_tracking_data3D(TEST_DIR[data_iter], SELECTED_TEST_DATA[data_iter], False)
original_A1 = test_data

MISSING_MAP_DIR = "./CMU/test_data_CMU_gap/10/10.txt"
missing_map, _ = read_tracking_data3D_without_RJ(MISSING_MAP_DIR, SELECTED_TEST_DATA[0])

A1 = original_A1 * missing_map
