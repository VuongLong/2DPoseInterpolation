# missing joint over a frame on whole frame
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
	print(A.shape)
	A_temp_zero = []

	for frame_index in range(A.shape[0]):
		A_temp_zero.append(get_random_joint_partially(A, arg.length, arg.missing_joint_partially, frame_index))

	for frame_index in range(A.shape[0]):
		print("current: ", frame_index)
		tmpA1 = []
		tmpA3 = []
		tmpA4 = []
		tmpA4_old = []
		tmpA3_old = []
		check_shift = True
		for x in range(1):
			A1 = np.copy(
				Tracking2D[arg.reference[0]+shift_A_value*2:arg.reference[0]+arg.length+shift_A_value*2])
			A1zero = np.copy(A1)
			A1zero[np.where(A_temp_zero[frame_index] == 0)] = 0

			# first formula
			# A1_star3, A0_star3,IUT,TTU1TA1R = interpolation_13(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
			# 													shift = check_shift, option = None)
			# tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

			# A1_star4, A0_star4,VTI,A1V1FR,A1_MeanMat = interpolation_24(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
			# 													shift = check_shift, option = None)
			# tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))

			# A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat)
			# tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))


			# 2nd formula
			# compute 1st method
			A1_star3, A0_star3,IUT,TTU1TA1R = interpolation_13_v2(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

			# compute 2nd method
			A1_star4, A0_star4,VTI,A1V1FR,A1_MeanMat = interpolation_24_v2(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))

			# compute 3th method
			# A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat)
			# tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))

			# compute 2 old methods
			A1_star3o, _,_,_ = interpolation_13(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA3_old.append(np.around(calculate_mse(A1, A1_star3o), decimals = 17))

			A1_star4o, _,_,_,_ = interpolation_24(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA4_old.append(np.around(calculate_mse(A1, A1_star4o), decimals = 17))
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

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	full_list = find_full_matrix(Tracking2D, 20)
	print(full_list)

	# process_hub(method = 3, joint = True)
	process_hub5(method = 5, joint = True)

	# target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	# contruct_skeletion_to_video(arg.input_dir, A1_star3, target, arg.output_dir, arg.output_video, arg.ingore_confidence)
	# show_video(arg.output_dir + '/' + arg.output_video, 200)
