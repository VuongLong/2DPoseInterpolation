import argparse

class arguments(argparse.Namespace):
	# Model arguments
	input_dir = './data2'
	output_dir = './output' 
	output_video = 'ChuckPangPadNha_2'
	# data_dir = './data1/04_0_BogMue.data' #'ChuckPangPadNha_2'
	data_dir = input_dir + '/' + output_video + ".data"
	
	length = 50
	AN_length = 150
	reference  = [0, 497]
	reference_task4 = [[0, 50], [50, 100], [100, 150]]
	shift_arr = [0, 1, 2, 3, 4, 5, length]
	missing_joint = [1, 2, 3, 4, 5]
	
	target = [0, 50]
	ingore_confidence = True

arg = arguments
