# implemention corresponds formula in section Yu - 08th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys


def process_hub5(method = 1, joint = True):
	resultA1 = []
	resultA3 = []
	resultA4 = []
	resultA4_old = []
	resultA3_old = []
	type_plot = "Frame"
	if joint:
		type_plot = "joint"
	shift_A_value = 23
	shift_A1_value = 10

	# A0's size is 2nxm
	# A_N is formed by pilling up A0 by N = N/m time.
	# A_N3 is formed by expanding along wide by K = N/m time
	A_N = np.array([])
	for x in arg.reference_task4_3D:
		tmp = np.copy(Tracking3D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)

	A_N3 = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.AN_length_3D])

	tmp_meanVec = np.mean(A_N3,0)
	A0_mean = np.tile(tmp_meanVec,(arg.length,1))
	AN3_MeanMat = np.tile(tmp_meanVec, (A_N3.shape[0], 1))
	tmp_meanVecUnit = np.tile(tmp_meanVec, (arg.length, 1)).T
	AN_MeanMat = np.tile(tmp_meanVecUnit, (3, 1)).T

	# option = [AN3_MeanMat, A0_mean] / None for task 3
	# option = [AN_MeanMat, A0_mean] / None for task 4

	A = np.copy(Tracking3D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length3D+shift_A_value])
	A_temp_zero = []
	for num_missing in arg.missing_number:
		if joint:
			# A_temp_zero.append(get_random_joint3D(A, arg.length3D, num_missing))
			A_temp_zero.append(get_remove_row3D(A, arg.length3D, num_missing))
		else:
			A_temp_zero.append(get_removed_peice3D(A, arg.length3D, num_missing))
	
	for current_frame_shift in range(20):
		print("current: ", current_frame_shift)
		tmpA1 = []
		tmpA3 = []
		tmpA4 = []
		tmpA4_old = []
		tmpA3_old = []
		check_shift = True
		if current_frame_shift == 0:
			check_shift = False
		for index_A_temp in range(len(arg.missing_number)):
			A1 = np.copy(
				Tracking3D[arg.reference[0]+shift_A_value+current_frame_shift*shift_A1_value:arg.reference[0]+arg.length3D+shift_A_value+current_frame_shift*shift_A1_value])
			A1zero = np.copy(A1)
			A1zero[np.where(A_temp_zero[index_A_temp] == 0)] = 0

			# compute T method
			A1_star3 = interpolation_13_v3(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None, Tmatrix = True)
			tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

			# compute F method
			A1_star4 = interpolation_24_v3(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None, Tmatrix = True)
			tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))

			# compute 2 old methods
			A1_star3o, _,_,_ = interpolation_13(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA3_old.append(np.around(calculate_mse(A1, A1_star3o), decimals = 17))

			A1_star4o, _,_,_,_ = interpolation_24(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA4_old.append(np.around(calculate_mse(A1, A1_star4o), decimals = 17))

		# resultA1.append(tmpA1)
		resultA3.append(tmpA3)
		resultA4.append(tmpA4)
		resultA3_old.append(tmpA3_old)
		resultA4_old.append(tmpA4_old)

	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	multi_export_xls(4, [resultA3, resultA4, resultA3_old, resultA4_old], file_name = file_name)
	# plot_line(resultA3, resultA4, file_name+"_cp34", type_plot, name1 = "Error T3", name2 = "Error T4", scale= shift_A1_value)
	plot_line3(resultA3_old, resultA3, resultA4, file_name+"_cp34", type_plot, scale= shift_A1_value)
	# plot_line(resultA1, resultA4, file_name+"_cp54", type_plot, name1 = "Error T5", name2 = "Error T4", scale= shift_A1_value)
	# plot_line(resultA1, resultA3, file_name+"_cp53", type_plot, name1 = "Error T5", name2 = "Error T3", scale= shift_A1_value)



if __name__ == '__main__':

	Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	Tracking3D = Tracking3D.astype(float)
	predicted_matrix = process_hub5(method = 5, joint = True)

	