# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys


def generate_missing_joint(n, m, number_missing):
	l = [x for x in range(n*m//3)]
	missing_joint = random.sample(l, number_missing)
	matrix = np.random.rand(n,m)
	for position in missing_joint:
		xx = position // m
		yy = position % m
		matrix[xx][yy] = 0
		matrix[xx][yy+1] = 0
		matrix[xx][yy+2] = 0
	return matrix

def process_hub5(method = 1, joint = True):
	resultA1 = []
	resultA3 = []
	resultA4 = []
	resultA4_old = []
	resultA3_old = []
	type_plot = "joint"

	A_N = np.array([])
	for x in arg.reference_task4_3D:
		tmp = np.copy(Tracking3D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)

	A_N3 = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.AN_length_3D])

	A = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.length3D])
	
	for percent_missing in range(3):
		print("current: ", (percent_missing+1)*10)
		tmpA3 = []
		tmpA4 = []
		check_shift = False
		random_starting_A1 = random.randint(1,1000)
		A1 = np.copy(
			Tracking3D[arg.reference[0]+random_starting_A1:arg.reference[0]+arg.length3D+random_starting_A1])
		missing_joint = A1.shape[0]*A1.shape[1] * (percent_missing+1) // 10
		missing_matrix = generate_missing_joint(A1.shape[0], A1.shape[1], missing_joint )

		A1zero = np.copy(A1)
		A1zero[np.where(missing_matrix == 0)] = missing_matrix[np.where(missing_matrix == 0)]
		print(A1zero.shape)
		np.savetxt("checkA1.txt", A1zero, fmt = "%.2f")
		# compute 1st method
		A1_star3 = interpolation_13_v2(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
															shift = check_shift, option = None)
		tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

		# compute 2nd method
		A1_star4 = interpolation_24_v2(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
															shift = check_shift, option = None)
		tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))


		# resultA1.append(tmpA1)
		resultA3.append(tmpA3)
		resultA4.append(tmpA4)

	file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	multi_export_xls(2, [resultA3, resultA4], file_name = file_name)
	plot_line3(resultA3, resultA3, resultA4, file_name+"_cp34", type_plot, scale= shift_A1_value)
	# plot_line(resultA1, resultA3, file_name+"_cp53", type_plot, name1 = "Error T5", name2 = "Error T3", scale= shift_A1_value)



if __name__ == '__main__':
	# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	Tracking3D, restore  = read_tracking_data3D_v2("./data3D/test.txt")
	Tracking3D = Tracking3D.astype(float)
	predicted_matrix = process_hub5(method = 5, joint = False)

	