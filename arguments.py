import argparse

class arguments(argparse.Namespace):
	# Model arguments
	input_dir = './data2'
	output_dir = './output' 
	output_video = 'ChuckPangPadNha_2'
	# data_dir = './data1/04_0_BogMue.data'
	data_dir = input_dir + '/' + output_video + ".data"
	#task 1 3
	length = 50
	AN_length = 50
	reference  = [0, 497]
	reference_task4 = [[0, length], [length+10, length*2+10], [length*3+10, length*4+10]]
	shift_arr = [0, 1, 2, 3, 4, 5, length]
	missing_joint = [1, 2, 3, 4, 5]
	#task 2 4
	# reference1  = [457, 507]
	# reference2  = [50, 100]
	
	target = [0, 50]
	ingore_confidence = True

arg = arguments
