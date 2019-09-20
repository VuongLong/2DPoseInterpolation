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
	# fastsong1_Take_001.bvh 'chuckpangpadNha_Take_001'
	data_dir3D = input_dir3D + '/' + output_video3D + ".bvh"
	new_dir3D = output_dir + '/bvh/' + output_video3D+ "_new" + ".bvh"
	# chuckpangpadNha_Take_001.bvh
	length = 60
	
	# AN_length = 300
	# reference  = [0, 497]
	# reference_task4 = [[0, 50], [50, 100], [100, 150],
	# 					[150, 200], [200, 250], [250, 300]]
	# AN_length = 180
	# reference  = [0, 497]
	# reference_task4 = [[0, 60], [60, 120], [120, 180]]

	# PLOS 1_3

	# length3D = 240
	# AN_length_3D = 2400
	# # reference_AN  = 1200
	# reference_task4_3D = [[0, 240], [240, 480], [480, 720], [720, 960], [960, 1200]]
	# reference_task4_3D_source = [[1200, 1440], [1440, 1680], [1680, 1920], [1920, 2160], [2160, 2400],[2400, 2640], [2640, 2880], [2880, 3120], [3120, 3360], [3360, 3600], [3600, 3840], [3840, 4080], [4080, 4320], [4320, 4560], [4560, 4800]]

	# PLOS 1_2	

	length3D = 280
	# AN_length_3D = 2400
	# # reference_AN  = 1200
	reference_task4_3D = [[0, 280], [280, 560], [560, 840], [840, 1120], [1120, 1400], [1400, 1680], [1680, 1960], [1960, 2240], [2240, 2520], [2520, 2800]]
	reference_task4_3D_source = [[2800, 3080], [3080, 3360], [3360, 3640], [3640, 3920], [3920, 4200], [4200, 4480], [4480, 4760], [4760, 5040]]


	shift_arr = [0, 1, 2, 3, 4, 5, length]
	missing_number = [1, 2]
	missing_joint_partially = 5

	target = [0, 50]
	ingore_confidence = True

	missing_row_arr = [2, 10, 1, 4, 5] # random select missing row

arg = arguments
