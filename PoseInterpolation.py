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


	shift_A_value = 10
	shift_A1_value = 10
	A_N = np.array([]) 
	for x in arg.reference_task4:
		tmp = np.copy(Tracking2D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)
	A_N = np.copy(Tracking2D[arg.reference[0]+20:arg.reference[0]+arg.AN_length]+20)

	
	A = np.copy(Tracking2D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length+shift_A_value]) 
	A_temp_zero = []
	for num_missing in arg.missing_joint:
		if joint:
			A_temp_zero.append(get_random_joint(A, arg.length, num_missing))
		else:
			A_temp_zero.append(get_removed_peice(A, arg.length, num_missing))

	for current_frame_shift in range (arg.length+1):
		tmpA1 = []
		tmpA0 = []	
		check_shift = True
		if current_frame_shift == 0:
			check_shift = False	
		for num_missing in arg.missing_joint:
			
			A1 = np.copy(
				Tracking2D[arg.reference[0]+current_frame_shift+shift_A1_value:arg.reference[0]+arg.length+current_frame_shift+shift_A1_value]) 

			A1zero = np.copy(A1)
			tmp = A_temp_zero[num_missing-1]
			A1zero[np.where(A_temp_zero[num_missing-1] == 0)] = 0

			# if joint:
			# 	A1zero = get_random_joint(A1, arg.length, num_missing)
			# else:
			# 	A1zero = get_removed_peice(A1, arg.length, num_missing)

			if method == 1:
				A1_star, A0_star,_,_ = interpolation_13(A, A ,A1zero, shift = check_shift)
			elif method == 2:
				A1_star, A0_star,_,_,_ = interpolation_24(A, A, A1zero, shift = check_shift)
			elif method == 3:
				A1_star, A0_star,_,_ = interpolation_13(A_N, A ,A1zero, shift = check_shift)
			elif method == 4:
				A1_star, A0_star,_,_,_ = interpolation_24(A_N, A ,A1zero, shift = check_shift)
			else:
				A1_star, A0_star,IUT,TTU1TA1R = interpolation_13(A_N, A ,A1zero, shift = check_shift)
				A1_star, A0_star,VTI,A1V1FR = interpolation_24(A_N, A ,A1zero, shift = check_shift)
				A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR, shift = check_shift)

			tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))
			tmpA0.append(np.around(calculate_mse(A, A0_star), decimals = 3))
			
		resultA1.append(tmpA1)
		resultA0.append(tmpA0)
	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.reference[0])
	export_xls(resultA0, resultA1, file_name = file_name)
	plot_line(resultA0, resultA1, file_name, type_plot)


def process_hub5(method = 1, joint = True):
	resultA1 = []
	resultA3 = []
	resultA4 = []
	resultA30 = []
	resultA40 = []
	type_plot = "Frame"
	if joint:
		type_plot = "Joint"
	shift_A_value = 0
	shift_A1_value = 10
	A_N = np.array([]) 
	for x in arg.reference_task4:
		tmp = np.copy(Tracking2D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)
	A_N3 = np.copy(Tracking2D[arg.reference[0]:arg.reference[0]+arg.AN_length])
	# A_N = np.copy(A_N3.T)
	A = np.copy(Tracking2D[arg.reference[0]+shift_A_value:arg.reference[0]+arg.length+shift_A_value]) 
	A_temp_zero = []
	for num_missing in arg.missing_joint:
		if joint:
			A_temp_zero.append(get_random_joint(A, arg.length, num_missing))
		else:
			A_temp_zero.append(get_removed_peice(A, arg.length, num_missing))

	for current_frame_shift in range(20):
		tmpA1 = []
		tmpA3 = []
		tmpA30 = []
		tmpA4 = []
		tmpA40 = []
		check_shift = True
		if current_frame_shift == 0:
			check_shift = False
		for num_missing in arg.missing_joint:
			A1 = np.copy(
				Tracking2D[arg.reference[0]+current_frame_shift*shift_A1_value:arg.reference[0]+arg.length+current_frame_shift*shift_A1_value]) 
			A1zero = np.copy(A1)
			tmp = A_temp_zero[num_missing-1]
			A1zero[np.where(A_temp_zero[num_missing-1] == 0)] = 0
			# if joint:
			# 	A1zero = get_random_joint(A1, arg.length, num_missing)
			# else:
			# 	A1zero = get_removed_peice(A1, arg.length, num_missing)
			
			A1_star3, A0_star3,IUT,TTU1TA1R = interpolation_13(np.copy(A_N3), np.copy(A) ,np.copy(A1zero), shift = check_shift)
			tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 3))
			# tmpA30.append(np.around(calculate_mse(A, A0_star3), decimals = 3))
			A1_star4, A0_star4,VTI,A1V1FR,A1_MeanMat = interpolation_24(np.copy(A_N), np.copy(A) ,np.copy(A1zero), shift = check_shift)
			tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 3))
			# tmpA40.append(np.around(calculate_mse(A, A0_star4), decimals = 3))
			# A1_star = interpolation(A1zero, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat)
			# tmpA1.append(np.around(calculate_mse(A1, A1_star), decimals = 3))
			
			

		# resultA1.append(tmpA1)
		resultA3.append(tmpA3)
		resultA4.append(tmpA4)
		# resultA30.append(tmpA30)
		# resultA40.append(tmpA40)

	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	# export_xls(resultA1, resultA3, resultA4, file_name = file_name)
	plot_line(resultA3, resultA4, file_name+"_cp34", type_plot, name1 = "Error T3", name2 = "Error T4", scale= shift_A1_value)
	# plot_line(resultA1, resultA4, file_name+"_cp54", type_plot, name1 = "Error T5", name2 = "Error T4", scale= shift_A1_value)
	# plot_line(resultA1, resultA3, file_name+"_cp53", type_plot, name1 = "Error T5", name2 = "Error T3", scale= shift_A1_value)

	# print("checker")
	# for i in range(5):
	# 	print(np.array(resultA1).T[i].mean())
	# 	print(np.array(resultA3).T[i].mean())
	# 	print(np.array(resultA4).T[i].mean())
	# print("end checker")
	

if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	full_list = find_full_matrix(Tracking2D, 20)
	print(full_list)
	
	# process_hub(method = 3, joint = True)
	process_hub5(method = 5, joint = True)

	#target = [arg.reference[0]+0, arg.reference[0]+arg.length+0]

	#contruct_skeletion_to_video(arg.input_dir, A1_star_13, target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	#show_video(arg.output_dir + '/' + arg.output_video, 200)	
