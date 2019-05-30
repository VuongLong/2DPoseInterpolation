import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys

def process_hub(method = 1, joint = True, type = "Frame"):
	resultA1 = []
	resultA0 = []
	shift_A_value = 0
	shift_A1_value = 0
	if method == 3:
		shift_A_value = 10
		shift_A1_value = 50
		A_N = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length])
	for current_frame_shift in range (arg.length+1):
		tmpA1 = []
		tmpA0 = []
		for num_missing in arg.missing_joint:

			A = np.copy(Tracking2D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length+shift_A_value]) 
			A1 = np.copy(
				Tracking2D[arg.reference[0]+current_frame_shift+shift_A1_value:arg.reference[0]+arg.length+current_frame_shift+shift_A1_value]) 


			if joint:
				A1zero = get_random_joint(A1, arg.length, num_missing)
			else:
				A1zero = get_removed_peice(A1, arg.length, num_missing)

			if method == 1:
				A1_star, A0_star = interpolation_13(A ,A1zero)
			elif method == 2:
				A1_star, A0_star = interpolation_24(A, A1zero)
			elif method == 3:
				A1_star, A0_star = interpolation_3(A_N, A ,A1zero)
			else:
				halt


			tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))
			tmpA0.append(np.around(calculate_mse(A, A0_star), decimals = 3))
			
		resultA1.append(tmpA1)
		resultA0.append(tmpA0)
	file_name = "Task"+str(method)+'_'+type+'_'+str(arg.length)+'_'+str(arg.reference[0])
	export_xls(resultA0, resultA1, file_name = file_name)
	plot_line(resultA0, resultA1, file_name, type)


if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	# full_list = find_full_matrix(Tracking2D, 20)
	# print(full_list)
	
	process_hub(method = 3, joint = False)

	#target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	#contruct_skeletion_to_video(arg.input_dir, A1_star_13, target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	#show_video(arg.output_dir + '/' + arg.output_video, 200)	
