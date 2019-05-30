import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys

def process_hub(method = 1, joint = True):
	resultA1 = []
	resultA0 = []
	shift_A_value = 0
	shift_A1_value = 0
	type_plot = "Frame"
	if joint:
		type_plot = "Joint"

	if method == 3:
		shift_A_value = 10
		shift_A1_value = 331
		A_N = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length])
	elif method == 4:
		shift_A_value = 10
		shift_A1_value = 331
		A_N = np.array([]) 
		for x in arg.reference_task4:
			tmp = np.copy(Tracking2D[x[0]:x[1]])
			if A_N.shape[0] != 0:
				A_N = np.concatenate((A_N, tmp), axis = 1)
			else:
				A_N = np.copy(tmp)
	elif method == 5:
		shift_A_value = 10
		shift_A1_value = 331
		A_N = np.array([]) 
		for x in arg.reference_task4:
			tmp = np.copy(Tracking2D[x[0]:x[1]])
			if A_N.shape[0] != 0:
				A_N = np.concatenate((A_N, tmp), axis = 1)
			else:
				A_N = np.copy(tmp)
		shift_A_value = 10
		shift_A1_value = 50
		A_N = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length])

	A = np.copy(Tracking2D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length+shift_A_value]) 
	for current_frame_shift in range (arg.length+1):
		tmpA1 = []
		tmpA0 = []
		for num_missing in arg.missing_joint:
			
			A1 = np.copy(
				Tracking2D[arg.reference[0]+current_frame_shift+shift_A1_value:arg.reference[0]+arg.length+current_frame_shift+shift_A1_value]) 

			if joint:
				A1zero = get_random_joint(A1, arg.length, num_missing)
			else:
				A1zero = get_removed_peice(A1, arg.length, num_missing)

			if method == 1:
				A1_star, A0_star,_,_ = interpolation_13(A, A ,A1zero)
			elif method == 2:
				A1_star, A0_star,_,_ = interpolation_24(A, A, A1zero)
			elif method == 3:
				A1_star, A0_star,_,_ = interpolation_13(A_N, A ,A1zero)
			elif method == 4:
				A1_star, A0_star,_,_ = interpolation_24(A_N, A ,A1zero)
			else:
				A1_star, A0_star,IUT,TTU1TA1R = interpolation_13(A_N, A ,A1zero)
				A1_star, A0_star,VTI,A1V1FR = interpolation_24(A_N, A ,A1zero)
				A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR)

			tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))
			tmpA0.append(np.around(calculate_mse(A, A0_star), decimals = 3))
			
		resultA1.append(tmpA1)
		resultA0.append(tmpA0)
	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.reference[0])
	export_xls(resultA0, resultA1, file_name = file_name)
	plot_line(resultA0, resultA1, file_name, type_plot)


if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	# full_list = find_full_matrix(Tracking2D, 20)
	# print(full_list)
	
	process_hub(method = 3, joint = True)


	#target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	#contruct_skeletion_to_video(arg.input_dir, A1_star_13, target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	#show_video(arg.output_dir + '/' + arg.output_video, 200)	
