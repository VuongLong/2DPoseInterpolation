# missing by random entries
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def generate_missing_joint(n, m, number_missing):
	m = m // 3
	l = [x for x in range(n*m)]
	missing_joint = random.sample(l, number_missing)
	matrix = np.ones((n,3*m))
	for position in missing_joint:
		xx = position // m
		yy = position % m
		matrix[xx][yy*3] = 0
		matrix[xx][yy*3+1] = 0
		matrix[xx][yy*3+2] = 0
	return matrix

def process_hub5(method = 1, joint = True, data = None):
	resultT = []
	resultF = []
	resultT1 = []
	resultF1 = []
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

	joint_percent = [1, 3, 5, 10]
	test_reference = arg.reference_task4_3D
	number_patch = len(arg.reference_task4_3D)
	sample = np.copy(Tracking3D[test_reference[0][0]:test_reference[0][1]])
	patch_missing = 0
	
	for joint in joint_percent:
		number_joint = int(joint*sample.shape[0]*sample.shape[1]/100)
		tmpT = []
		tmpF = []
		tmpT1 = []
		tmpF1 = []
		for times in range(20):
			full_matrix = np.ones(Tracking3D[0:test_reference[number_patch-1][1]].shape)
			patch_arr = [0]*number_patch
			patch_arr[patch_missing] = 1
			print(patch_arr)
			print("frame missing: ", joint, "times: ",times)

			A_N = A_N_source
			A_N3 = A_N3_source
			for x in range(number_patch):
				if patch_arr[x] > 0:
					# generate missing matrix
					missing_matrix = generate_missing_joint(
						sample.shape[0], sample.shape[1], number_joint)		
						
					full_matrix[test_reference[x][0]:test_reference[x][0]+arg.length3D] = missing_matrix
						# fetch the rest of patch for reference AN and AN3
				else:
					print("patch add to refer: ", x)
					tmp = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
					A_N = np.hstack((A_N, tmp))
					A_N3 = np.vstack((A_N3, tmp))
			print("reference A_N update missing data: ",A_N.shape)
			print("reference A_N3 update missing data: ",A_N3.shape)
			# interpolation for each patch
			tmpT = []
			tmpF = []
			tmpT1 = []
			tmpF1 = []
			for x in range(number_patch):
				if patch_arr[x] > 0:
					# get data which corespond to starting frame of A1
					A1 = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
					missing_matrix = full_matrix[test_reference[x][0]:test_reference[x][1]]
					A1zero = np.copy(A1)
					A1zero[np.where(missing_matrix == 0)] = 0
					np.savetxt("original.txt", A1, fmt = "%.2f")
					np.savetxt("zero.txt", A1zero, fmt = "%.2f")


					# T method
					A1_starT = interpolation_13_v6(np.copy(A_N3),np.copy(A1zero))
					tmpT.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_starT[np.where(A1zero == 0)]), decimals = 17))
					np.savetxt("recover.txt", A1_starT, fmt = "%.2f")
					# F method
					A1_starF = interpolation_24_v6(np.copy(A_N),np.copy(A1zero), Tracking3D)
					tmpF.append(np.around(calculate_mae_matrix(
						A1[np.where(A1zero == 0)]- A1_starF[np.where(A1zero == 0)]), decimals = 17))

					
			tmpT.append(np.asarray(tmpT).sum())
			tmpF.append(np.asarray(tmpF).sum())
			
		resultT.append(np.asarray(tmpT).mean())
		resultF.append(np.asarray(tmpF).mean())
		
	return resultT, resultF



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
		rT, rF = process_hub5(method = 5, joint = True, data = None)
		result.append([rT, rF])
	for x in range(len(result)):
		print(result[x])
