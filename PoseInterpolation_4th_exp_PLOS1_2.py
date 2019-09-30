# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def generate_missing_joint(n, m, frame_length, starting_frame):
	frames = int(frame_length * 120)
	matrix = np.ones((n,m))
	joint_in = []
	counter = 0
	number_joint = 3
	Long_matrix = []
	while counter < number_joint:
		counter+=1
		tmp = random.randint(1, m//3-3)
		while tmp in joint_in:
			tmp = random.randint(1, m//3-3)

		start_missing_frame = random.randint(1, n-frames)
		missing_joint = tmp
		# print("start_missing_frame: ", start_missing_frame, "joint: ", missing_joint)
		for frame in range(start_missing_frame, start_missing_frame+frames):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
			Long_matrix.append([frame+starting_frame, missing_joint*3])
			Long_matrix.append([frame+starting_frame, missing_joint*3+1])
			Long_matrix.append([frame+starting_frame, missing_joint*3+2])
	return matrix, np.asarray(Long_matrix)



def process_hub5(method = 1, joint = True, data = None):
	resultA3 = []
	resultA4 = []
	list_patch = arg.reference_task4_3D_source
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

	test_patch = [2, 4, 6, 8, 10]
	test_reference = arg.reference_task4_3D
	number_patch = len(arg.reference_task4_3D)
	patch_A1_in_refer = 1
	starting_frame_A1 = test_reference[patch_A1_in_refer][0]
	for patch in test_patch:
		lmiss = 1
		tmpA3 = []
		tmpA4 = []

		for times in range(20):
			print("current: ", patch, times)
			# select random patch in refer for A1
			
			# get data which corespond to starting frame of A1
			A1 = np.copy(Tracking3D[starting_frame_A1:starting_frame_A1+arg.length3D])

			# generate missing matrix
			missing_matrix, Long_matrix = generate_missing_joint(A1.shape[0], A1.shape[1], lmiss, starting_frame_A1)
			full_matrix = np.ones(Tracking3D[0:test_reference[patch-1][1]].shape)
			A1zero = np.copy(A1)
			A1zero[np.where(missing_matrix == 0)] = 0
			full_matrix[starting_frame_A1:arg.length3D+starting_frame_A1] = missing_matrix
			np.savetxt("./test_data1/"+ str(patch*2) +"/"+str(times)+ ".txt", full_matrix, fmt = "%d")
			np.savetxt("./test_data1/"+ str(patch*2) +"/"+str(times)+ "_map.txt", Long_matrix, fmt = "%d")
			# fetch the rest of patch for reference AN and AN3
			A_N = A_N_source
			A_N3 = A_N3_source
			for unused_patch in range(patch):
				if unused_patch != patch_A1_in_refer:
					tmp = np.copy(Tracking3D[test_reference[unused_patch][0]:test_reference[unused_patch][1]])
					A_N = np.hstack((A_N, tmp))
					A_N3 = np.vstack((A_N3, tmp))
			print("reference A_N: ",A_N.shape)
			print("reference A_N3: ",A_N3.shape)
			A1_star3 = interpolation_13_v6(np.copy(A_N3),np.copy(A1zero), Tracking3D)
			tmpA3.append(np.around(calculate_mae_matrix(A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)]), decimals = 17))
			# compute 2nd method
			A1_star4 = interpolation_24_v6(np.copy(A_N),np.copy(A1zero), Tracking3D)
			tmpA4.append(np.around(calculate_mae_matrix(A1[np.where(A1zero == 0)]- A1_star4[np.where(A1zero == 0)]), decimals = 17))

		Tresult = np.asarray(tmpA3).mean()
		Fresult = np.asarray(tmpA4).mean()
		# resultA1.append(tmpA1)
		resultA3.append(Tresult)
		resultA4.append(Fresult)
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
		AN_source = np.hstack(
			[np.copy(source[list_patch[i][0]:list_patch[i][1]]) for i in range(K)])
		tmp_AN.append(AN_source)
		AN3_source = np.copy(source[list_patch[0][0]: list_patch[-1][1]])
		tmp_AN3.append(AN3_source)
	source_AN = np.hstack(tmp_AN)
	source_AN3 = np.vstack(tmp_AN3)

	print("reference source:")
	print(source_AN.shape)
	print(source_AN3.shape)

	data_link = ["./data3D/135_02.txt"]
	# data_link = ["./data3D/135_02.txt","./data3D/85_12.txt", "./data3D/HDM_mm_02-02_02_120.txt", "./data3D/HDM_mm_01-02_03_120.txt", "./data3D/HDM_mm_03-02_01_120.txt"]
	result = []
	for x in data_link:
		print(x)
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
		Tracking3D, restore  = read_tracking_data3D_v2(x)
		Tracking3D = Tracking3D.astype(float)
		r3, r4 = process_hub5(method = 5, joint = True, data = [source_AN, source_AN3])
		result.append([r3,r4])
	for x in range(len(result)):
		print(result[x])
