# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def load_missing():
	link = "./test_data/120/0.txt"
	matrix = []
	f=open(link, 'r')
	for line in f:
		elements = line[:-1].split(' ')
		matrix.append(list(map(int, elements)))
	f.close()

	matrix = np.array(matrix) # list can not read by index while arr can be
	return matrix

def process_hub5(method = 1, joint = True, data = None):
	resultA3 = []
	resultA4 = []
	list_patch = arg.reference_task4_3D_source
	print(list_patch)
	A_N_source = np.hstack(
		[np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
	A_N3_source = np.copy(Tracking3D[list_patch[0][0]: list_patch[-1][1]])
	print("original data reference A_N: ",A_N_source.shape)
	print("original data reference A_N3: ",A_N3_source.shape)
	if data != None:
		A_N_source = np.hstack((A_N_source, data[0]))
		A_N3_source = np.vstack((A_N3_source, data[1]))
	print("update reference:")
	print("reference A_N: ",A_N_source.shape)
	print("reference A_N3: ",A_N3_source.shape)

	length_missing = [0.5]
	test_reference = arg.reference_task4_3D
	number_patch = len(arg.reference_task4_3D)
	sample = np.copy(Tracking3D[test_reference[0][0]:test_reference[0][1]])

	patch_arr = [0] * number_patch
	patch_arr[4] = 1
	full_matrix = load_missing()
	A_N = A_N_source
	A_N3 = A_N3_source
	for x in range(number_patch):
		if patch_arr[x] == 0:
			tmp = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
			A_N = np.hstack((A_N, tmp))
			A_N3 = np.vstack((A_N3, tmp))
	# interpolation for each patch
	tmpT = []
	tmpF = []
	for x in range(number_patch):
		if patch_arr[x] > 0:
			# get data which corespond to starting frame of A1
			A1 = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
			missing_matrix = full_matrix[test_reference[x][0]:test_reference[x][1]]
			A1zero = np.copy(A1)
			A1zero[np.where(missing_matrix == 0)] = 0

			A1_star3 = interpolation_13_v6(np.copy(A_N3),np.copy(A1zero), Tracking3D)
			tmpT.append(np.around(calculate_mae_matrix(
				A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)]), decimals = 17))
			# compute 2nd method
			A1_star4 = interpolation_24_v6(np.copy(A_N),np.copy(A1zero), Tracking3D)
			np.savetxt("missing.txt", A1zero, fmt = "%.3f", delimiter=', ')
			np.savetxt("result.txt", A1_star4, fmt = "%.3f", delimiter=', ')
			np.savetxt("original.txt", A1, fmt = "%.3f", delimiter=', ')
			tmpF.append(np.around(calculate_mae_matrix(
				A1[np.where(A1zero == 0)]- A1_star4[np.where(A1zero == 0)]), decimals = 17))
	resultA3 = np.asarray(tmpT).sum()
	resultA4 = np.asarray(tmpF).sum()

	return resultA3, resultA4



if __name__ == '__main__':

	refer_link = ["./data3D/135_01.txt", "./data3D/135_03.txt"]
	tmp_AN = []
	tmp_AN3= []
	for x in refer_link:
		print("reading source: ", x)
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
		source , _  = read_tracking_data3D_v2(x)
		source = source.astype(float)
		K = source.shape[0] // arg.length3D
		list_patch = [[x*arg.length3D, (x+1)*arg.length3D] for x in range(K)]
		A_N_source = np.hstack(
			[np.copy(source[list_patch[i][0]:list_patch[i][1]]) for i in range(K)])
		tmp_AN.append(A_N_source)
		A_N3_source = np.copy(source[list_patch[0][0]: list_patch[-1][1]])
		tmp_AN3.append(A_N3_source)
	source_AN = np.hstack(tmp_AN)
	source_AN3 = np.vstack(tmp_AN3)

	print("reference source:")
	print(source_AN.shape)
	print(source_AN3.shape)

	data_link = "./data3D/135_02.txt"
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	Tracking3D, _  = read_tracking_data3D_v2(data_link)
	Tracking3D = Tracking3D.astype(float)
	r3, r4 = process_hub5(method = 5, joint = True, data = [source_AN, source_AN3])
	print(r3, r4)