import argparse

class arguments(argparse.Namespace):
	# Model arguments 2D
	input_dir = './data2'
	output_dir = './output'
	output_video = 'ChuckPangPadNha_2'
	# data_dir = './data1/04_0_BogMue.data' #'ChuckPangPadNha_2' "9_Fast Song 05"
	data_dir = input_dir + '/' + output_video + ".data"


	input_dir3D = './data3D'
	output_video3D = 'chuckpangpadNha_Take_001'
	data_dir3D = input_dir3D + '/' + output_video3D + ".bvh"
	new_dir3D = output_dir + '/' + output_video3D+ "_new" + ".bvh"
	# chuckpangpadNha_Take_001.bvh
	length = 50
	length3D = 45
	# AN_length = 300
	# reference  = [0, 497]
	# reference_task4 = [[0, 50], [50, 100], [100, 150],
	# 					[150, 200], [200, 250], [250, 300]]
	AN_length = 150
	reference  = [0, 497]
	reference_task4 = [[0, 50], [50, 100], [100, 150]]

	AN_length_3D = 135
	reference  = [0, 497]
	reference_task4 = [[0, 45], [45, 90], [90, 135]]


	shift_arr = [0, 1, 2, 3, 4, 5, length]
	missing_number = [1, 2, 3, 4, 5]
	missing_joint_partially = 10

	target = [0, 50]
	ingore_confidence = True

	missing_row_arr = [14, 10, 1, 4, 5] # random select missing row

arg = arguments
