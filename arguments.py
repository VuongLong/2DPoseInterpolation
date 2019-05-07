import argparse

class arguments(argparse.Namespace):
	# Model arguments
	input_dir = './data1'
	output_dir = './output' 
	output_video = '04_0_BogMue'
	data_dir = './data1/04_0_BogMue.data'
	#task 1 3
	reference  = [0, 200]

	#task 2 4
	reference1  = [0, 100]
	reference2  = [100, 200]
	
	target = [0, 100]
	ingore_confidence = True

arg = arguments
