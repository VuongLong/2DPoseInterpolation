# implemention corresponds formula in section Yu - 04th 07 2019
# exp 2: with respect to a set of frames, please test from missing 1 joint, 2 joints… to the last one
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import matplotlib.pyplot as plt

def generate_missing_joint_frame(n, m, frames, number_missing):
	m = m // 3
	frames_missing_number = n // 10 * frames
	list_frame = [x for x in range(n)]
	missing_frame = random.sample(list_frame, frames_missing_number)
	matrix = np.random.rand(n,m*3)
	for x in missing_frame:
		list_joint = [x for x in range(m//3)]
		missing_joint = random.sampel(list_joint, number_missing)
		for joint in missing_joint:
			matrix[x][joint*3] = 0
			matrix[x][joint*3+1] = 0
			matrix[x][joint*3+2] = 0
	return matrix

def process_hub5(method = 1, joint = True):
	
	type_plot = "joint"

	A_N = np.array([])
	for x in arg.reference_task4_3D:
		tmp = np.copy(Tracking3D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)

	A_N3 = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.AN_length_3D])

	#A = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.length3D])
	resultA3 = []
	resultA4 = []	
	frame_missing = 3
	for number_joint in range(Tracking3D.shape[1]//3):
		tmpA3 = []
		tmpA4 = []
		for time in range(10):
		
			#print("current: ", (percent_missing+1)*10, times)
			
			check_shift = False
			random_starting_A1 = random.randint(1,400)
			A1 = np.copy(
				Tracking3D[arg.reference[0]+random_starting_A1:arg.reference[0]+arg.length3D+random_starting_A1])
			A = np.copy(A1)

			missing_matrix = generate_missing_joint_frame(A1.shape[0], A1.shape[1], frame_missing, number_joint)

			A1zero = np.copy(A1)
			A1zero[np.where(missing_matrix == 0)] = missing_matrix[np.where(missing_matrix == 0)]
			# print("A goc truoc khi tru mean",A1[np.where(A1zero == 0)])
			# print(A1zero.shape)
			# compute 1st method
			A1_star3 = interpolation_13_v2(np.copy(A_N3), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))

			# compute 2nd method
			A1_star4 = interpolation_24_v2(np.copy(A_N), np.copy(A) ,np.copy(A1zero),
																shift = check_shift, option = None)
			tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))

		resultA3.append(np.asarray(tmpA3).mean())
		resultA4.append(np.asarray(tmpA4).mean())

	return resultA3, resultA4

	# file_name = "Task"+str(method)+'_'+type_plot+'_'+str(arg.length)+'_'+str(arg.AN_length)
	# multi_export_xls(2, [resultA3, resultA4], file_name = file_name)
	#plot_line3(resultA3, resultA3, resultA4, file_name+"_cp34", type_plot, scale= 1)
	# plot_line(resultA1, resultA3, file_name+"_cp53", type_plot, name1 = "Error T5", name2 = "Error T3", scale= shift_A1_value)



if __name__ == '__main__':
	data_link = ["./data3D/85_02.txt", "./data3D/85_12.txt", "./data3D/135_02.txt", "./data3D/HDM_mm_02-02_02_120.txt", "./data3D/HDM_mm_01-02_03_120.txt", "./data3D/HDM_mm_03-02_01_120.txt"]
	counter = 0
	result = []
	for x in data_link:
		print(x)
		counter += 1
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
		Tracking3D, restore  = read_tracking_data3D_v2(x)
		#Tracking3D, restore  = read_tracking_data3D_v2("./data3D/85_12.txt")
		Tracking3D = Tracking3D.astype(float)
		r1, r2 = predicted_matrix = process_hub5(method = 5, joint = True)
		result.append([r1, r2])

	title = "Missing with respect to a set of frames"

	fig = plt.figure()
	fig.suptitle(title, fontsize=16)

	ax1 = plt.subplot("611")
	ax1.set_ylabel(data_link[0])

	ax2 = plt.subplot("612")
	ax2.set_ylabel(data_link[1])

	ax3 = plt.subplot("613")
	ax3.set_ylabel(data_link[2])

	ax4 = plt.subplot("614")
	ax4.set_ylabel(data_link[3])


	ax5 = plt.subplot("615")
	ax5.set_ylabel(data_link[4])

	ax6 = plt.subplot("616")
	ax6.set_xlabel('number joint missing')
	# ax4.set_xlabel('Joint index')
	# ax4.set_xlabel('Frame index')
	ax6.set_ylabel(data_link[5])


	for i in range(2):
		d1 = result[0][i]
		d2 = result[1][i]
		d3 = result[2][i]
		d4 = result[3][i]
		d5 = result[4][i]
		d6 = result[5][i]

		#print(data)
		ax1.plot(np.arange(len(d1)), d1, color=color[i], linewidth=1)

		ax2.plot(np.arange(len(d2)), d2, color=color[i], linewidth=1)

		ax3.plot(np.arange(len(d3)), d3, color=color[i], linewidth=1)

		ax4.plot(np.arange(len(d4)), d4, color=color[i], linewidth=1)

		ax5.plot(np.arange(len(d5)), d5, color=color[i], linewidth=1)

		ax6.plot(np.arange(len(d6)), d6, color=color[i], linewidth=1)

	plt.legend(loc=2, prop={'size': 10})
	figure = plt.gcf() # get current figure
	figure.set_size_inches(6, 8)
	plt.savefig('./plot_output/'+title+'.png', dpi=600)

