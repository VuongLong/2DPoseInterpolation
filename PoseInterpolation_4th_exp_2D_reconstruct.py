# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys


def process_hub5(method = 1, joint = True, data = None):
	list_patch = arg.reference_2D_source
	A_NF_source = np.hstack(
		[np.copy(Tracking2D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
	A_NT_source = np.copy(Tracking2D[list_patch[0][0]: list_patch[-1][1]])
	print("original data reference A_N: ",A_NF_source.shape)
	print("original data reference A_N3: ",A_NT_source.shape)
	if data != None:
		A_NF_source = np.hstack((A_NF_source, data[0]))
		A_NT_source = np.vstack((A_NT_source, data[1]))
	print("update reference:")
	print("reference A_N: ",A_NF_source.shape)
	print("reference A_N3: ",A_NT_source.shape)
	test_data = np.copy(Tracking2D[arg.test_2D[0][0]: arg.test_2D[0][1]])
	number_patch = test_data.shape[0] // arg.length2D

	A_NF = A_NF_source
	A_NT = A_NT_source
	counter = 0
	shift = (arg.length2D//2)
	l = - shift
	while 1 > 0:
		counter += 1
		print(counter)
		# compute T method
		l += shift
		r = l + arg.length2D
		if r > test_data.shape[0]:
			r = test_data.shape[0]
			l = r - arg.length2D
		A1zero = np.copy(test_data[l:r])
		A1_star3 = interpolation_13_v6(np.copy(A_NT), np.copy(A1zero) ,Tracking2D)
		test_data[l:r] = A1_star3
		if r == test_data.shape[0]:
			break
	Tracking2D[arg.test_2D[0][0]: arg.test_2D[0][1]] = test_data
	np.savetxt("reconstruct_2D.txt", Tracking2D, fmt = "%.2f")
	# compute F method
	# A1_star4 = interpolation_24_v6(np.copy(A_NF), np.copy(A1zero) ,Tracking2D)
	contruct_skeletion_to_video('./data2D', Tracking2D, arg.test_2D[0], arg.output_dir, "reconstruct_2D", arg.ingore_confidence)
	print(calculate_mse_matrix_Yu(Tracking2D - full_matrix, missing_joint))

if __name__ == '__main__':

	# refer_link = ["./data2D/outputChaimue.data"]
	# tmp_ANF = []
	# tmp_ANT= []
	# for x in refer_link:
	# 	print("reading source: ", x)
	# 	source  = read_tracking_data(x, arg.ingore_confidence)
	# 	source = source.astype(float)
	# 	K = source.shape[0] // arg.length2D
	# 	list_patch = [[x*arg.length2D, (x+1)*arg.length2D] for x in range(K)]
	# 	ANF_source = np.hstack(
	# 		[np.copy(source[list_patch[i][0]:list_patch[i][1]]) for i in range(K)])
	# 	tmp_ANF.append(ANF_source)
	# 	ANT_source = np.copy(source[list_patch[0][0]: list_patch[-1][1]])
	# 	tmp_ANT.append(ANT_source)
	# source_ANF = np.hstack(tmp_ANF)
	# source_ANT = np.vstack(tmp_ANT)

	# print("reference source:")
	# print(source_ANF.shape)
	# print(source_ANT.shape)

	Tracking2D = read_tracking_data("./data2D/outputChaimue.data", arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	contruct_skeletion_to_video('./data2D', Tracking2D, arg.test_2D[0], arg.output_dir, "missing_2D", arg.ingore_confidence)
	missing_joint = 0
	for x in range(Tracking2D.shape[0]):
		for y in range(Tracking2D.shape[1]):
			if Tracking2D[x][y] == 0:
				missing_joint += 1
	print(missing_joint)
	full_matrix = read_tracking_data("./data2D/outputChaimue_full.data", arg.ingore_confidence)
	full_matrix = full_matrix.astype(float)
	process_hub5(method = 5, joint = True, data = None)


	