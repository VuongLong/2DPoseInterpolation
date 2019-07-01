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
	A_temp_zero = []
	# for common missing joint
	#for num_missing in arg.missing_number:
	#	if joint:
	#		A_temp_zero.append(get_random_joint(A, arg.length, num_missing))
	#		# A_temp_zero.append(get_remove_row(A, arg.length, num_missing))
	#	else:
	#		A_temp_zero.append(get_removed_peice(A, arg.length, num_missing))
	# for partially missing joint

	for frame_index in range(A.shape[1]):
		A_temp_zero.append(get_random_joint_partially(A, arg.length, arg.missing_joint_partially, frame_index))

	current_frame_shift = 1
	for frame_index in range(A.shape[1]):
		tmpA1 = []
		tmpA3 = []
		tmpA30 = []
		tmpA4 = []
		tmpA40 = []
		check_shift = True
		if current_frame_shift == 0:
			check_shift = False
		for x in range(1):
			A1 = np.copy(
				Tracking2D[arg.reference[0]+shift_A_value*2+current_frame_shift*shift_A1_value:arg.reference[0]+arg.length+shift_A_value*2+current_frame_shift*shift_A1_value]) 
			A1zero = np.copy(A1)
			A1zero[np.where(A_temp_zero[frame_index] == 0)] = 0
	
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

	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	export_xls(resultA1, resultA3, resultA4, file_name = file_name)
	#plot_line(resultA3, resultA4, file_name+"_cp34", type_plot, name1 = "Error T3", name2 = "Error T4", scale= shift_A1_value)
	plot_line3(resultA1, resultA3, resultA4, file_name+"_cp34", type_plot, scale= shift_A1_value)
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
