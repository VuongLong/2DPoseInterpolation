import argparse

class arguments(argparse.Namespace):
	# Model arguments
	input_dir = './data2'
	output_dir = './output' 
	output_video = 'ChuckPangPadNha_2'
	# data_dir = './data1/04_0_BogMue.data'
	data_dir = input_dir + '/' + output_video + ".data"
	#task 1 3
	reference  = [0, 507]
	length = 6
	shift_arr = [0, 1, 2, 3, 4, 5, length]
	missing_joint = [1, 2, 3, 4, 5]
	#task 2 4
	# reference1  = [457, 507]
	# reference2  = [50, 100]
	
	target = [0, 50]
	ingore_confidence = True

arg = arguments
