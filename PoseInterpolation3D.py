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
	resultA30 = []
	resultA40 = []
	type_plot = "Frame"
	if joint:
		type_plot = "Joint"
	shift_A_value = 23
	shift_A1_value = 1
	A_N = np.array([])
	for x in arg.reference_task4_3D:
		tmp = np.copy(Tracking3D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)

	A_N3 = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.AN_length_3D])

	tmp_meanVec = np.mean(A_N3,0)
	A0_mean = np.tile(tmp_meanVec,(arg.length3D,1))
	AN3_MeanMat = np.tile(tmp_meanVec, (A_N3.shape[0], 1))
	tmp_meanVecUnit = np.tile(tmp_meanVec, (arg.length3D, 1)).T
	AN_MeanMat = np.tile(tmp_meanVecUnit, (3, 1)).T

	# option = [AN3_MeanMat, A0_mean] / None for task 3
	# option = [AN_MeanMat, A0_mean] / None for task 4

	A = np.copy(Tracking3D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length3D+shift_A_value])
	A_temp_zero = []
	
	for num_missing in range(A.shape[1]):
		A_temp_zero.append(get_random_joint_partially3D(A, arg.length3D, arg.missing_joint_partially,num_missing))

	counter = 0
	for num_missing in range(A.shape[1]):
		tmpA1 = []
		tmpA3 = []
		tmpA30 = []
		tmpA4 = []
		tmpA40 = []
		check_shift = True
		for xx in range(1):
			A1 = np.copy(
				Tracking3D[arg.reference[0]+shift_A_value*15:arg.reference[0]+arg.length3D+shift_A_value*15])
			A1zero = np.copy(A1)
			A1zero[np.where(A_temp_zero[num_missing] == 0)] = 0
			counter += 1
			A1_star3, A0_star3,IUT,TTU1TA1R = interpolation_13(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

			A1_star4, A0_star4,VTI,A1V1FR,A1_MeanMat = interpolation_24(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))
			A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat)
			tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))

		resultA1.append(tmpA1)
		resultA3.append(tmpA3)
		resultA4.append(tmpA4)
	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length3D)+'_'+str(arg.AN_length_3D)
	export_xls(resultA1, resultA3, resultA4, file_name = file_name)
	#plot_line(resultA3, resultA4, file_name+"_cp34", type_plot, name1 = "Error T3", name2 = "Error T4", scale= shift_A1_value)
	plot_line3(resultA1, resultA3, resultA4, file_name+"_cp34", type_plot, scale= shift_A1_value)
	# plot_line(resultA1, resultA4, file_name+"_cp54", type_plot, name1 = "Error T5", name2 = "Error T4", scale= shift_A1_value)
	# plot_line(resultA1, resultA3, file_name+"_cp53", type_plot, name1 = "Error T5", name2 = "Error T3", scale= shift_A1_value)



if __name__ == '__main__':

	Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	Tracking3D = Tracking3D.astype(float)
	process_hub5(method = 5, joint = True)

	# reconstruct file

