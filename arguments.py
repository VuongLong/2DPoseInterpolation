import argparse

class arguments(argparse.Namespace):
	# Model arguments 2D
	input_dir = './data2'
	output_dir = './output'
	output_video = 'ChuckPangPadNha_2'
	# data_dir = './data1/04_0_BogMue.data' #'ChuckPangPadNha_2' "9_Fast Song 05"
	data_dir = input_dir + '/' + output_video + ".data"


	input_dir3D = './data3D'
	output_video3D = 'ChaiMue_Take_001'
	# fastsong1_Take_001.bvh 'chuckpangpadNha_Take_001'
	data_dir3D = input_dir3D + '/' + output_video3D + ".bvh"
	new_dir3D = output_dir + '/bvh/' + output_video3D+ "_new" + ".bvh"
	# chuckpangpadNha_Take_001.bvh
	length2D = 100
	reference_2D_source = [[0, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600], 
	[600, 700], [700, 800], [800, 900], [900, 1000], [1000, 1100], [1100, 1200], [1200, 1300], 
	[1300, 1400], [1400, 1500], [1500, 1600], [1600, 1700], [1700, 1800], [1800, 1900], [1900, 2000], 
	[2000, 2100], [2100, 2200], [2200, 2300], [2300, 2400], [2400, 2500], [2500, 2600], [2600, 2700]]
	test_2D = [[2700, 2997]]
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

	# length3D = 280
	# # AN_length_3D = 2400
	# # # reference_AN  = 1200
	# reference_task4_3D = [[0, 280], [280, 560], [560, 840], [840, 1120], [1120, 1400], [1400, 1680], [1680, 1960], [1960, 2240], [2240, 2520], [2520, 2800]]
	# reference_task4_3D_source = [[0, 280], [2240, 2520]]

	# CMU _ WALK
	# length3D = 130
	# # AN_length_3D = 2400
	# # # reference_AN  = 1200
	# reference_task4_3D = [[0, 130]]
	# reference_task4_3D_source = [[130, 260]]

	# Li's test
	length3D = 100
	# AN_length_3D = 2400
	# # reference_AN  = 1200
	reference_task4_3D = [[450, 550]]
	reference_task4_3D_source = [[50, 150], [150, 250], [250, 350], [350, 450], [550, 650], [650, 750], [750, 850]]

	cheat_array = [0, 1, 2, 3, 14, 10, 6, 9]

	shift_arr = [0, 1, 2, 3, 4, 5]
	missing_number = [1, 2]
	missing_joint_partially = 5

	target = [0, 50]
	ingore_confidence = True

	missing_row_arr = [2, 10, 1, 4, 5] # random select missing row

arg = arguments
