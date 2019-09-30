# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys


def generate_missing_joint(n, m, number_missing):
	print(n, m)
	l = [x for x in range(n*m//2)]
	missing_joint = random.sample(l, number_missing)
	matrix = np.random.rand(n,m)
	for position in missing_joint:
		xx = position // (m//2)
		yy = position % (m//2)
		matrix[xx][yy*2] = 0
		matrix[xx][yy*2+1] = 0
	return matrix

def process_hub5(method = 1, joint = True, data = None):
	resultT = []
	resultF = []
	list_patch = arg.reference_2D_source
	print(list_patch)
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
	percent_missing = 1
	tmpT = []
	tmpF = []
	test_reference = arg.test_2D
	number_patch = len(test_reference)
	for times in range(20):
		print("current: ", percent_missing*10," time: " ,times)
		

		random_starting_A1 = random.randint(0,test_reference[-1][1])
		A1 = np.copy(Tracking2D[random_starting_A1:arg.length2D+random_starting_A1])
		missing_joint = A1.shape[0]*A1.shape[1] * percent_missing // 30
		missing_matrix = generate_missing_joint(A1.shape[0], A1.shape[1], missing_joint )
		A1zero = np.copy(A1)
		A1zero[np.where(missing_matrix == 0)] = missing_matrix[np.where(missing_matrix == 0)]
		print("A1 start from: ",random_starting_A1)

		A_NF = A_NF_source
		A_NT = A_NT_source
		print(test_reference)
		for x in range(number_patch):
			if (test_reference[x][0] > random_starting_A1+arg.length2D) or \
			(test_reference[x][1] < random_starting_A1):
				tmp = np.copy(Tracking2D[test_reference[x][0]: test_reference[x][1]])
				A_NT = np.vstack((A_NT, tmp))
				A_NF = np.hstack((A_NF, tmp))
		

		# compute T method
		A1_star3 = interpolation_13_v6(np.copy(A_NT), np.copy(A1zero) ,Tracking2D)													
		tmpT.append(np.around(calculate_mse_matrix_Yu(
			A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)], missing_joint), decimals = 17))

		# compute F method
		A1_star4 = interpolation_24_v6(np.copy(A_NF), np.copy(A1zero) ,Tracking2D)
		tmpF.append(np.around(calculate_mse_matrix_Yu(
			A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)], missing_joint), decimals = 17))

	# resultA1.append(tmpA1)
	resultT.append(np.asarray(tmpT).mean())
	resultF.append(np.asarray(tmpF).mean())

	# file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	# multi_export_xls(2, [resultA3, resultA4], file_name = file_name)
	# plot_line3(resultA3, resultA3, resultA4, file_name+"_cp34", type_plot, scale= shift_A1_value)
	# plot_line(resultA1, resultA3, file_name+"_cp53", type_plot, name1 = "Error T5", name2 = "Error T3", scale= shift_A1_value)
	return [resultT, resultF]


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
	r3, r4 = process_hub5(method = 5, joint = True, data = None)

	print("result", r3, r4)

	