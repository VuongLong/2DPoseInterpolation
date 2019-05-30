import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg


def process_hub(method = 1, joint = True):
	resultA1 = []
	resultA0 = []
	for current_frame_shift in range (arg.length+1):
		tmpA1 = []
		tmpA0 = []
		for num_missing in arg.missing_joint:

			A = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.length]) 
			A1 = np.copy(Tracking2D[arg.reference[0]+current_frame_shift:arg.reference[0]+arg.length+current_frame_shift]) 

			if joint:
				A1zero = get_random_joint(A1, arg.length, num_missing)
			else:
				A1zero = get_removed_peice(A1, arg.length, num_missing)

			if method == 1:
				A1_star, A0_star = interpolation_13(A ,A1zero)
			elif method == 2:
				A1_star, A0_star = interpolation_24(A, A1zero)

			tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))
			tmpA0.append(np.around(calculate_mse(A, A0_star), decimals = 3))
			
		resultA1.append(tmpA1)
		resultA0.append(tmpA0)

	target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	#contruct_skeletion_to_video(arg.input_dir, A1_star_13, target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	#show_video(arg.output_dir + '/' + arg.output_video, 200)

	export_xls(resultA0, resultA1)
	plot_line(resultA0, resultA1, "Task2")


if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	full_list = find_full_matrix(Tracking2D, 20)
	# print(full_list)
	# method 1
	process_hub(method = 2, joint = False)
