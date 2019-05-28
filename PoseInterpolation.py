import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys

if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	full_list = find_full_matrix(Tracking2D, 50)
	# print(full_list)
	# method 1
	M1_resultA1 = []
	M1_resultA0 = []
	# method 2
	#M2_resultA1 = []
	#M2_resultA0 = []
	A_N = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length]) 	# this code for task 3 method 1
	for current_frame_shift in range (arg.length+1):
		M1_tmpA1 = []
		M1_tmpA0 = []
		#M2_tmpA1 = []
		#M2_tmpA0 = []
		for num_missing in arg.missing_joint:
			# dim1 = frame, dim2 = joint
			#A = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.length]) # this code for task 2, the below one is using for task3
			A = np.copy(Tracking2D[arg.reference[0]+10:arg.reference[0]+arg.length+10])
			#A1 = np.copy(Tracking2D[arg.reference[0]+current_frame_shift:arg.reference[0]+arg.length+current_frame_shift]) # this code for task 2, the below one is using for task3
			A1 = np.copy(Tracking2D[arg.reference[0]+current_frame_shift+50:arg.reference[0]+arg.length+current_frame_shift+50])

			A1zero = get_random_joint(A1, arg.length, num_missing)
			# A1zero = get_removed_peice(A1, arg.length, num_missing)


			#A1_star_13, A0_star_13 = interpolation_13(A ,A1zero)
			A1_star_13, A0_star_13 = interpolation_3(A_N, A ,A1zero)
			M1_tmpA1.append(np.around(calculate_mse(A1, A1_star_13), decimals = 3))
			M1_tmpA0.append(np.around(calculate_mse(A, A0_star_13), decimals = 3))
			#A1_star_24, A0_star_24 = interpolation_24(A, A1zero)
			#M2_tmpA1.append(np.around(calculate_mse(A1, A1_star_13), decimals = 3))
			#M2_tmpA0.append(np.around(calculate_mse(A, A0_star_13), decimals = 3))
			
		M1_resultA1.append(M1_tmpA1)
		M1_resultA0.append(M1_tmpA0)
		#M2_resultA1.append(M2_tmpA1)
		#M2_resultA0.append(M2_tmpA0)
	# print('M11', M1_resultA1, 'M10', M1_resultA0)
	#print('M21', M2_resultA1, 'M20', M2_resultA0)

	#target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	#contruct_skeletion_to_video(arg.input_dir, A1_star_13, target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	#show_video(arg.output_dir + '/' + arg.output_video, 200)

	#export_xls(M1_resultA0, M1_resultA1, M2_resultA0, M2_resultA1)
	export_xls(M1_resultA0, M1_resultA1, file_name = "Task3_joint_70_20")
	plot_line(M1_resultA0, M1_resultA1, "Task3_joint_70_20")
	#plot_line(M2_resultA0, M2_resultA1, "Task2")


