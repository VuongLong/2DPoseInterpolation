# joint over frame
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random

def process_hub5():
	resultA1 = []
	resultA3 = []
	resultA4 = []
	resultA30 = []
	resultA40 = []
	type_plot = "Joint over a Frame"
	shift_A_value = 23
	shift_A1_value = 1 # must greater or equal to 1
	A_N = np.array([])
	for x in arg.reference_task4:
		tmp = np.copy(Tracking2D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)
	A_N3 = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length])


	tmp_meanVec = np.mean(A_N3,0)
	A0_mean = np.tile(tmp_meanVec,(arg.length,1))
	AN3_MeanMat = np.tile(tmp_meanVec, (A_N3.shape[0], 1))
	tmp_meanVecUnit = np.tile(tmp_meanVec, (arg.length, 1)).T
	AN_MeanMat = np.tile(tmp_meanVecUnit, (3, 1)).T

	# option = [AN3_MeanMat, A0_mean] / None for task 3
	# option = [AN_MeanMat, A0_mean] / None for task 4

	A = np.copy(Tracking2D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length+shift_A_value])
	A_temp_zero = []

	frame_index = random.randint(0,A.shape[0])
	for joint in range(25):
		A_temp_zero.append(get_joint_over_Aframe(A, joint+1, frame_index))

	current_frame_shift = 1
	for joint in range(25):
		tmpA1 = []
		tmpA3 = []
		tmpA30 = []
		tmpA4 = []
		tmpA40 = []
		check_shift = True
		# check shift == true mean A1 is same with A
		for x in range(1):
			A1 = np.copy(
				Tracking2D[arg.reference[0]+shift_A_value*2:arg.reference[0]+arg.length+shift_A_value*2])
			A1zero = np.copy(A1)
			A1zero[np.where(A_temp_zero[joint] == 0)] = 0

			A1_star3, A0_star3,IUT,TTU1TA1R = interpolation_13(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None, Tmatrix = True)
			tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

			A1_star4, A0_star4,VTI,A1V1FR,A1_MeanMat = interpolation_24(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None, Tmatrix = True)
			tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))

			# A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat)
			# tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))

		# resultA1.append(tmpA1)
		resultA3.append(tmpA3)
		resultA4.append(tmpA4)

	file_name = type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	# export_xls(resultA1, resultA3, resultA4, file_name = file_name)
	plot_line(resultA3, resultA4, file_name+"_cp34", type_plot, name1 = "Error T3", name2 = "Error T4", scale= shift_A1_value)
	# plot_line3(resultA1, resultA3, resultA4, file_name+"_cp34", type_plot, scale= shift_A1_value)



if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	full_list = find_full_matrix(Tracking2D, 20)
	print(full_list)

	process_hub5()
