import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg

if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	print(Tracking2D.shape)
	full_list = find_full_matrix(Tracking2D, 20)
	# print(full_list)
	# method 1

	M1_resultA1 = []
	M1_resultA0 = []
	# method 2
	# M2_resultA1 = []
	# M2_resultA0 = []
	# A_N = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length]) 	# this code for task 3 method 1
	
	A_N = np.array([]) 	# this code for task 4 method 2
	for x in arg.reference_task4:
		tmp = np.copy(Tracking2D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 0)
		else:
			A_N = np.copy(tmp)

	# A = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.length]) # this code for task 1&2, the below one is using for task3
	# A = np.copy(Tracking2D[arg.reference[0]+5:arg.reference[0]+arg.length+5])  # this code for task 3, the below one is uisng for task 4
	A = np.copy(Tracking2D[arg.reference[0]+10:arg.reference[0]+arg.length+10]) # this code for task 4
	for current_frame_shift in range (arg.length+1):
		M1_tmpA1 = []
		M1_tmpA0 = []
		# M2_tmpA1 = []
		# M2_tmpA0 = []
		for num_missing in arg.missing_joint:
			A1 = np.copy(Tracking2D[arg.reference[0]+current_frame_shift+331:arg.reference[0]+arg.length+current_frame_shift+331]) # this code for task 1&2, the below one is using for task3
			# A1 = np.copy(Tracking2D[arg.reference[0]+current_frame_shift+5:arg.reference[0]+arg.length+current_frame_shift+5])

			A1zero = get_random_joint(A1, arg.length, num_missing)
			# A1zero = get_removed_peice(A1, arg.length, num_missing)

			A1_star_13, A0_star_13 = interpolation_4(A_N, A ,A1zero)
			M1_tmpA1.append(np.around(calculate_mse(A1, A1_star_13), decimals = 3))
			M1_tmpA0.append(np.around(calculate_mse(A, A0_star_13), decimals = 3))
			# A1_star_24, A0_star_24 = interpolation_24(A, A1zero)
			# M2_tmpA1.append(np.around(calculate_mse(A1, A1_star_13), decimals = 3))
			# M2_tmpA0.append(np.around(calculate_mse(A, A0_star_13), decimals = 3))
			
		M1_resultA1.append(M1_tmpA1)
		M1_resultA0.append(M1_tmpA0)
		# M2_resultA1.append(M2_tmpA1)
		# M2_resultA0.append(M2_tmpA0)
	print(np.array(M1_resultA1).shape)
	print('M11', M1_resultA1, 'M10', M1_resultA0)
	# print('M21', M2_resultA1, 'M20', M2_resultA0)

	target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	#contruct_skeletion_to_video(arg.input_dir, A1_star_13, target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	#show_video(arg.output_dir + '/' + arg.output_video, 200)

	# export_xls(M1_resultA0, M1_resultA1, M2_resultA0, M2_resultA1)
	export_xls(M1_resultA0, M1_resultA1, file_name = "Task4_joint")
	plot_line(M1_resultA0, M1_resultA1, "Task4_joint")
	# plot_line(M2_resultA0, M2_resultA1, "Task2")


