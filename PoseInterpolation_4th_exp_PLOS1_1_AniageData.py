# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def generate_missing_joint(n, m, frame_length, number_gap):
	matrix = np.ones((n,m))
	counter = 0
	joint_in = []
	while counter < number_gap:
		counter+=1
		tmp = random.randint(0, m//3-3)
		while tmp in joint_in:
			tmp = random.randint(0, m//3-3)
			joint_in.append(tmp)
		start_missing_frame = np.random.randint(0, n - frame_length)
		missing_joint = tmp
		# print("start_missing_frame: ", start_missing_frame, "joint: ", missing_joint)
		for frame in range(start_missing_frame, start_missing_frame+frame_length):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
	return matrix



def process_hub5(method = 1, joint = True, data = None):
	resultA3 = []
	resultA4 = []
	resultA5 = []
	resultA6 = []
	resultA7 = []
	list_patch = arg.reference_task4_3D_source
	list_patch = arg.reference_task4_3D_source
	if len(list_patch) > 0:
		print(list_patch)
		A_N_source = np.hstack(
			[np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
		A_N3_source = np.vstack(
			[np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
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

	length_missing = [10]
	# length_missing = [10, 20, 50, 70]
	test_reference = arg.reference_task4_3D
	number_patch = len(arg.reference_task4_3D)
	sample = np.copy(Tracking3D[test_reference[0][0]:test_reference[0][1]])
	patch_missing = 0
	
	for lmiss in length_missing:
		nframe = lmiss
		tmpA3 = []
		tmpA4 = []
		tmpA5 = []
		tmpA6 = []
		tmpA7 = []
		for times in range(10):
			print("current: ", lmiss, times)
			# patch_arr = [0]*number_patch
			# missing_gap_arr = []
			# shuffle_arr = [x for x in range(number_patch)]
			# random.shuffle(shuffle_arr)
			# counter = 1
			# for x in range(number_patch):
			# 	tmp = random.randint(0,counter)
			# 	counter -= tmp
			# 	missing_gap_arr.append(tmp)
			# counter = 0
			# for x in range(number_patch-1):
			# 	counter += missing_gap_arr[x]
			# missing_gap_arr[-1] = 1 - counter
			# for x in range(number_patch):
			# 	patch_arr[shuffle_arr[x]] = missing_gap_arr[x]

			full_matrix = np.ones(Tracking3D[0:test_reference[number_patch-1][1]].shape)
			patch_arr = [0]*number_patch
			patch_arr[patch_missing] = 1
			print(patch_arr)

			A_N = A_N_source
			A_N3 = A_N3_source
			for x in range(number_patch):
				starting_frame_A1 = test_reference[x][0]
				if patch_arr[x] > 0:
					print("patch add to missing: ", x)
					# generate missing matrix
					missing_matrix = generate_missing_joint(
						sample.shape[0], sample.shape[1], lmiss, 3)		
						
					full_matrix[starting_frame_A1:arg.length3D+starting_frame_A1] = missing_matrix
						# fetch the rest of patch for reference AN and AN3
				else:
					print("patch add to refer: ", x)
					tmp = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
					A_N = np.hstack((A_N, tmp))
					A_N3 = np.vstack((A_N3, tmp))
			print("reference A_N update missing data: ",A_N.shape)
			print("reference A_N3 update missing data: ",A_N3.shape)
			# np.savetxt("./test_data_Aniage/"+ str(nframe) +"/"+str(times)+ ".txt", full_matrix, fmt = "%d")
			# np.savetxt("./test_data/"+ str(nframe) +"/"+str(times)+ "_patch.txt", np.asarray(patch_arr), fmt = "%d")
			# interpolation for each patch
			tmpT = []
			tmpF = []
			tmpG = []
			tmpH = []
			tmpJ = []
			for x in range(number_patch):
				if patch_arr[x] > 0:
					# get data which corespond to starting frame of A1
					A1 = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
					missing_matrix = full_matrix[test_reference[x][0]:test_reference[x][1]]
					A1zero = np.copy(A1)
					A1zero[np.where(missing_matrix == 0)] = 0
					# np.savetxt("original.txt", A1, fmt = "%.2f")
					# np.savetxt("zero.txt", A1zero, fmt = "%.2f")

					A1_star4 = interpolation_13_v7(np.copy(A1zero), np.copy(A1zero))
					tmpF.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_star4[np.where(A1zero == 0)]), decimals = 17))
					

					A1_star3 = interpolation_13_v6(np.copy(A_N3),np.copy(A1zero))
					tmpT.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)]), decimals = 17))
					# np.savetxt("recover.txt", A1_star3, fmt = "%.2f")

					

					A1_star5 = interpolation_13_v2(np.vstack((A_N3, A1zero)), np.copy(A1zero))
					tmpG.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_star5[np.where(A1zero == 0)]), decimals = 17))

					# compute 2nd method
					A1_star6 = interpolation_24_v6(np.copy(A_N),np.copy(A1zero))
					tmpH.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_star6[np.where(A1zero == 0)]), decimals = 17))

					A1_star7 = interpolation_24_v6_v2(np.copy(A_N),np.copy(A1zero))
					tmpJ.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_star7[np.where(A1zero == 0)]), decimals = 17))


			tmpA3.append(np.asarray(tmpT).sum())
			tmpA4.append(np.asarray(tmpF).sum())
			tmpA5.append(np.asarray(tmpG).sum())
			tmpA6.append(np.asarray(tmpH).sum())
			tmpA7.append(np.asarray(tmpJ).sum())

		resultA3.append(np.asarray(tmpA3).mean())
		resultA4.append(np.asarray(tmpA4).mean())
		resultA5.append(np.asarray(tmpA5).mean())
		resultA6.append(np.asarray(tmpA6).mean())
		resultA7.append(np.asarray(tmpA7).mean())
	return resultA3, resultA4, resultA5, resultA6, resultA7



if __name__ == '__main__':

	# refer_link = ["./data3D/data/08_01.txt", "./data3D/data/07_01.txt"]
	# tmp_AN = []
	# tmp_AN3= []
	# for x in refer_link:
	# 	print("reading source: ", x)
	# 	# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	# 	source , _  = read_tracking_data3D_v2(x)
	# 	source = source.astype(float)
	# 	K = source.shape[0] // arg.length3D
	# 	list_patch = [[x*arg.length3D, (x+1)*arg.length3D] for x in range(K)]
	# 	A_N_source = np.hstack(
	# 		[np.copy(source[list_patch[i][0]:list_patch[i][1]]) for i in range(K)])
	# 	tmp_AN.append(A_N_source)
	# 	A_N3_source = np.copy(source[list_patch[0][0]: list_patch[-1][1]])
	# 	tmp_AN3.append(A_N3_source)
	# source_AN = np.hstack(tmp_AN)
	# source_AN3 = np.vstack(tmp_AN3)

	# print("reference source:")
	# print(source_AN.shape)
	# print(source_AN3.shape)

	data_link = ["./data3D/fastsong7.txt"]
	# data_link = ["./data3D/135_02.txt","./data3D/85_12.txt", "./data3D/HDM_mm_02-02_02_120.txt", "./data3D/HDM_mm_01-02_03_120.txt", "./data3D/HDM_mm_03-02_01_120.txt"]
	result = []
	for x in data_link:
		print(x)
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
		Tracking3D, _  = read_tracking_data3D_v2(x)
		Tracking3D = Tracking3D.astype(float)
		r3, r4, r5, r6, r7 = process_hub5(method = 5, joint = True, data = None)
		result.append([r3,r4, r5, r6, r7])
	for x in range(len(result)):
		print(result[x])
