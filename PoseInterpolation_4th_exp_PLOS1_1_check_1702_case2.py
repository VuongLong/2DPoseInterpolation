# compare with PCA or PLOS 1 with normalization
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random
from Yu_new_02.main import *
import os

def load_missing(sub_link = None):
	if sub_link == None:
		link = "./test_data_Aniage_gap/5/1.txt"
	else:
		link = sub_link
	matrix = []
	f=open(link, 'r')
	print(link)
	for line in f:
		elements = line[:-1].split(' ')
		matrix.append(list(map(int, elements)))
	f.close()

	matrix = np.array(matrix) # list can not read by index while arr can be
	return matrix
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

def create_folder_result(link):
	try:
	    os.mkdir(link)
	except OSError:
	    print ("Creation of the directory %s failed" % link)
	else:
	    print ("Successfully created the directory %s " % link)

def process_hub5(data = None):
	resultA3 = []
	resultA4 = []
	resultA5 = []
	resultA6 = []
	list_patch = arg.reference_task4_3D_source
	print(list_patch)
	A_N_source = np.hstack(
		[np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
	A_N3_source = np.vstack(
			[np.copy(Tracking3D[list_patch[i][0]:list_patch[i][1]]) for i in range(len(list_patch))])
	print("original data reference A_N: ",A_N_source.shape)
	print("original data reference A_N3: ",A_N3_source.shape)
	if data != None:
		A_N_source_added = np.hstack(( data[0], A_N_source))
		A_N3_source_added = np.vstack((data[1], A_N3_source))
	else:
		A_N_source_added = A_N_source
		A_N3_source_added = A_N3_source
	print("update reference:")
	print("reference A_N: ",A_N_source_added.shape)
	print("reference A_N3: ",A_N3_source_added.shape)
	test_folder = "./test_only_1/test/"
	# test_folder = "./fastsong7/test_data_Aniage_/"
	# test_folder = "./test_data_Aniage_leng/"
	# test_folder = "./test_data_CMU_gap/"
	order_fol = []
	for test_name in os.listdir(test_folder):
		current_folder = test_folder + test_name
		if os.path.isdir(current_folder):
			order_fol.append(test_name)
			tmpA3 = []
			tmpA4 = []
			tmpA5 = []
			tmpA6 = []
			test_reference = arg.reference_task4_3D
			number_patch = len(arg.reference_task4_3D)
			sample = np.copy(Tracking3D[test_reference[0][0]:test_reference[0][1]])

			patch_arr = [0] * number_patch
			patch_arr[0] = 1
			A_N = A_N_source
			A_N3 = A_N3_source
			number_test = 20
			for sub_test in range(number_test):
				result_path = current_folder+'/'+str(sub_test) + ".txt"
				print(result_path)
				# if os.path.isdir(current_folder+'/'+sub_test) :
				tmpT = []
				tmpF = []
				tmpG = []
				tmpV = []
				# full_matrix = load_missing()
				full_matrix = load_missing(result_path)
				for x in range(number_patch):
					if patch_arr[x] > 0:
						# get data which corespond to starting frame of A1
						A1 = np.copy(Tracking3D[test_reference[x][0]:test_reference[x][1]])
						missing_matrix = full_matrix[test_reference[x][0]:test_reference[x][1]]
						A1zero = np.copy(A1)
						A1zero[np.where(missing_matrix == 0)] = 0
						# tmp = np.vstack((A_N3_source_added, A1))
						# np.savetxt("data_ANIAGE.txt", tmp, fmt = "%.2f")
						# stop
						# np.savetxt("checkA1origin.txt", A1, fmt = "%.3f")
						A1_star3 = interpolation_weighted_dang_v2(np.copy(A_N3_source_added), np.copy(A1zero))
						value = np.around(calculate_mae_matrix(
							A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)]), decimals = 17)
						print(value)
						tmpT.append(value)
						# np.savetxt("A1_star3.txt", A1_star3, fmt = "%.3f")
						# np.savetxt("A1zero.txt", A1zero, fmt = "%.3f")

						# A1_star4 = PCA_PLOS1_F4(np.copy(A_N3_source_added), np.copy(A1zero))
						# # A1_star7 = PCA_PLOS1(np.copy(A1zero), np.copy(A1zero))
						# tmpF.append(np.around(calculate_mae_matrix(
						# 	A1[np.where(A1zero == 0)]- A1_star4[np.where(A1zero == 0)]), decimals = 17))

					
						# A1_star5 = interpolation_weighted_T_1702(np.copy(A_N3_source_added), np.copy(A1zero), True)
						# tmpG.append(np.around(calculate_mae_matrix(
						# 	A1[np.where(A1zero == 0)]- A1_star5[np.where(A1zero == 0)]), decimals = 17))

						# A1_star6 = interpolation_weighted_dang(np.copy(A_N3_source_added), np.copy(A1zero))
						# tmpV.append(np.around(calculate_mae_matrix(
						# 	A1[np.where(A1zero == 0)]- A1_star6[np.where(A1zero == 0)]), decimals = 17))
						# # save file for rendering
						#np.savetxt(result_path + "/original.txt", A1, fmt = "%.2f")
						#np.savetxt(result_path + "/PCA.txt", A1_star7, fmt = "%.2f")
						#np.savetxt(result_path + "/our_method.txt", A1_star8, fmt = "%.2f")
				tmpA3.append(np.asarray(tmpT).sum())
				tmpA4.append(np.asarray(tmpF).sum())
				tmpA5.append(np.asarray(tmpG).sum())
				tmpA6.append(np.asarray(tmpV).sum())
			resultA3.append(np.asarray(tmpA3).mean())
			resultA4.append(np.asarray(tmpA4).mean())
			resultA5.append(np.asarray(tmpA5).mean())
			resultA6.append(np.asarray(tmpA6).mean())
			break
	print(order_fol)
	return [resultA3, resultA4, resultA5, resultA6]



if __name__ == '__main__':

	# refer_link = ["./data3D/fastsong6.txt",]
	# resource_refer = [[50, 250]]
	# # refer_link = ["./data3D/fastsong7.txt"]
	# # resource_refer = [[450, 750]]
	# tmp_AN = []
	# tmp_AN3= []
	# counter = 0
	# for x in refer_link:
	# 	print("reading source: ", x)
	# 	# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	# 	source , _  = read_tracking_data3D_v2(x)
	# 	source = remove_joint(source)
	# 	source = source.astype(float)
	# 	source = source[resource_refer[counter][0]:resource_refer[counter][1]]
	# 	counter += 1
	# 	K = source.shape[0] // arg.length3D
	# 	list_patch = [[x*arg.length3D, (x+1)*arg.length3D] for x in range(K)]
	# 	AN_source = np.hstack(
	# 		[np.copy(source[list_patch[i][0]:list_patch[i][1]]) for i in range(K)])
	# 	tmp_AN.append(AN_source)
	# 	AN3_source = np.copy(source[list_patch[0][0]: list_patch[-1][1]])
	# 	tmp_AN3.append(AN3_source)
	# source_AN = np.hstack(tmp_AN)
	# source_AN3 = np.vstack(tmp_AN3)

	# print("reference source:")
	# print(source_AN.shape)
	# print(source_AN3.shape)

	data_link = "./data3D/fastsong7.txt"
	# data_link = "./data3D/135_02.txt"
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	Tracking3D, _  = read_tracking_data3D_v2(data_link)
	Tracking3D = remove_joint(Tracking3D)
	Tracking3D = Tracking3D.astype(float)
	# result = process_hub5(data = [source_AN, source_AN3])
	result = process_hub5()
	print(result)