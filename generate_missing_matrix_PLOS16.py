# generate missing gap in specific joint according to PLOS16
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random
from datetime import datetime

def generate_missing_joint_gap(n, m, frame_length, number_gap):
	# CMU: LSHO 9; LKNE: 28; LWA 14

	# ANIAGE: LSHO 2; LKNE 10; LWA 4
	frames = 70
	matrix = np.ones((n,m))
	joints = np.arange(m//3)
	counter = 0
	while counter < number_gap:
		counter+=1
		# tmp = arg.cheat_array[counter-1]
		missing_joint = 14
		start_missing_frame = random.randint(1, n-frames)
		# print("start_missing_frame: ", start_missing_frame, "joint: ", missing_joint)
		for frame in range(start_missing_frame, start_missing_frame+frames):
			matrix[frame, missing_joint*3] = 0
			matrix[frame, missing_joint*3+1] = 0
			matrix[frame, missing_joint*3+2] = 0
	counter = 0
	for x in range(n):
		for y in range(m):
			if matrix[x][y] == 0: counter +=1
	print("percent missing: ", 100 * counter / (n*m))
	return matrix



def process_gap_missing():
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


	print("reference A_N: ",A_N_source.shape)
	print("reference A_N3: ",A_N3_source.shape)

	gaps = [1]
	# gaps = [1, 3, 5, 9, 12]
	test_reference = arg.reference_task4_3D
	number_patch = len(arg.reference_task4_3D)
	sample = np.copy(Tracking3D[test_reference[0][0]:test_reference[0][1]])
	patch_missing = 0

	for gap in gaps:
		lmiss = 1
		for times in range(20):
			print("current: ", gap, times)

			# patch_arr = [0]*number_patch
			# tmp_missing_gap = []
			# counter = gap
			# shuffle_arr = [x for x in range(number_patch)]
			# random.shuffle(shuffle_arr)
			# for x in range(number_patch):
			# 	tmp = 10
			# 	while tmp >= 10:
			# 		tmp = random.randint(0,counter)
			# 	counter -= tmp
			# 	tmp_missing_gap.append(tmp)
			# counter = 0
			# for x in range(number_patch):
			# 	patch_arr[shuffle_arr[x]] = tmp_missing_gap[x]
			# for x in range(number_patch-1):
			# 	counter += patch_arr[x]
			# patch_arr[-1] = gap - counter
			

			full_matrix = np.ones(Tracking3D[0:test_reference[number_patch-1][1]].shape)
			patch_arr = [0]*number_patch
			patch_arr[patch_missing] = gap
			print(patch_arr)

			for x in range(number_patch):
				if patch_arr[x] > 0:
					starting_frame_A1 = test_reference[x][0]
					# generate missing matrix
					missing_matrix = generate_missing_joint_gap(
						sample.shape[0], sample.shape[1], lmiss, patch_arr[x])		
					full_matrix[starting_frame_A1:arg.length3D+starting_frame_A1] = missing_matrix

			# np.savetxt("./Aniage_gap_specific_joint/"+str(times)+ ".txt", full_matrix, fmt = "%d")
			np.savetxt("./CMU_gap_specific_joint/"+str(times)+ ".txt", full_matrix, fmt = "%d")

	# f = open("./Aniage_gap_specific_joint/info.txt", "w")
	f = open("./CMU_gap_specific_joint/info.txt", "w")
	f.write(str(datetime.now()))
	f.close()		
	return 


def remove_joint(data):
	list_del = []
	list_del_joint = [5, 9, 14, 18]

	for x in list_del_joint:
		list_del.append(x*3)
		list_del.append(x*3+1)
		list_del.append(x*3+2)
	data = np.delete(data, list_del, 1)
	print(data.shape)
	return data 

if __name__ == '__main__':

	# data_link = ["./data3D/fastsong7.txt"]
	data_link = ["./data3D/135_02.txt"]
	Tracking3D, _  = read_tracking_data3D_v2(data_link[0])
	# Tracking3D = remove_joint(Tracking3D)
	Tracking3D = Tracking3D.astype(float)
	process_gap_missing()
