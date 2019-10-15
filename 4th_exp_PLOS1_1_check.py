# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def generate_missing_joint(n, m, frame_length, number_gap, starting_frame):
	frames = int(frame_length * 120)
	matrix = np.ones((n,m))
	counter = 0
	joint_in = []
	print(number_gap)
	number_joint = number_gap
	while counter < number_joint:
		counter+=1
		tmp = random.randint(0, m//3-3)
		while tmp in joint_in:
			tmp = random.randint(0, m//3-3)
		start_missing_frame = random.randint(1, n-frames)
		missing_joint = tmp
		# print("start_missing_frame: ", start_missing_frame, "joint: ", missing_joint)
		for frame in range(start_missing_frame, start_missing_frame+frames):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
	return matrix


def check_full_matrix(matrix):
	tmp = matrix.shape[0] * matrix.shape[1]
	if tmp != np.sum(matrix):
		return False
	return True




def process_hub5(method = 1, joint = True, data = None):
	resultA3 = []
	resultA4 = []
	list_patch = arg.reference_task4_3D_source
	if len(list_patch) > 0:
		print(list_patch)
		A_N_source = np.hstack(
			[np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
		A_N3_source = np.copy(Tracking3D[list_patch[0][0]: list_patch[-1][1]])
		print("original data reference A_N: ",A_N_source.shape)
		print("original data reference A_N3: ",A_N3_source.shape)
	else:
		A_N_source = data[0]
		A_N3_source = data[1]

	if data != None:
		A_N_source = np.hstack((A_N_source, data[0]))
		A_N3_source = np.vstack((A_N3_source, data[1]))
	print("update reference:")
	print("reference A_N: ",A_N_source.shape)
	print("reference A_N3: ",A_N3_source.shape)

	length_missing = [0.5, 1, 2]
	test_reference = arg.reference_task4_3D
	number_patch = len(arg.reference_task4_3D)
	sample = np.copy(Tracking3D[test_reference[0][0]:test_reference[0][1]])
	for lmiss in length_missing:
		nframe = int(lmiss*120)
		tmpA3 = []
		tmpA4 = []
		for times in range(20):
			print("current: ", lmiss, times)
			current_link = "./test_data/"+ str(nframe) +"/"+str(times)+ ".txt"
			tmp_matrix, _ = read_tracking_data3D_v3(current_link)
			full_matrix = tmp_matrix.astype(int)
			patch_arr = [0]*number_patch

			A_N = A_N_source
			A_N3 = A_N3_source
			for x in range(number_patch):
				l = test_reference[x][0]
				r = test_reference[x][1]
				tmp_matrix = full_matrix[l:r]
				if check_full_matrix(tmp_matrix):
					tmp = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
					A_N = np.hstack((A_N, tmp))
					A_N3 = np.vstack((A_N3, tmp))
				else:
					patch_arr[x] = 1
			print("reference A_N update missing data: ",A_N.shape)
			print("reference A_N3 update missing data: ",A_N3.shape)
			# interpolation for each patch
			print(patch_arr)
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
					tmpF.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_star4[np.where(A1zero == 0)]), decimals = 17))
			tmpA3.append(np.asarray(tmpT).sum())
			tmpA4.append(np.asarray(tmpF).sum())

		resultA3.append(np.asarray(tmpA3).mean())
		resultA4.append(np.asarray(tmpA4).mean())
	return resultA3, resultA4



if __name__ == '__main__':

	refer_link = ["./data3D/data/07_06.txt", "./data3D/data/07_08.txt"]
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

	data_link = ["./data3D/data/07_02.txt"]
	# data_link = ["./data3D/135_02.txt","./data3D/85_12.txt", "./data3D/HDM_mm_02-02_02_120.txt", "./data3D/HDM_mm_01-02_03_120.txt", "./data3D/HDM_mm_03-02_01_120.txt"]
	result = []
	for x in data_link:
		print(x)
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
		Tracking3D, _  = read_tracking_data3D_v2(x)
		Tracking3D = Tracking3D.astype(float)
		r3, r4 = process_hub5(method = 5, joint = True, data = None)
		result.append([r3,r4])
	for x in range(len(result)):
		print(result[x])
