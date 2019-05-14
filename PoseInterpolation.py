import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg

if __name__ == '__main__':

	Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
	Tracking2D = Tracking2D.astype(float)
	full_list = find_full_matrix(Tracking2D, 50)
	print(full_list)
	
	###### Task 1-3 ########
	A = Tracking2D[arg.reference[0]:arg.reference[1]].T
	A1 = Tracking2D[arg.target[0]:arg.target[1]].T
	
	A_star_13, IUT, TTU1TA1R = interpolation_13(A, A1)

	########################

	###### Task 2-4 ########
	Ar1 = Tracking2D[arg.reference1[0]:arg.reference1[1]].T
	Ar2 = Tracking2D[arg.reference2[0]:arg.reference2[1]].T
	A = np.concatenate((Ar1, Ar2), axis=0)
	A1 = Tracking2D[arg.target[0]:arg.target[1]].T

	A_star_24, VTI, A1V1FR = interpolation_24(A, A1)
	########################
	
	###### Task  5  ########
	A_star_5 = interpolation(A1, IUT, TTU1TA1R, VTI, A1V1FR)
	#print(A_star_5.shape)	
	########################
	
	contruct_skeletion_to_video(arg.input_dir, A_star_5, arg.target, arg.output_dir, arg.output_video, arg.ingore_confidence)	
	show_video(arg.output_dir + '/' + arg.output_video, 200)
