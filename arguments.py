import argparse

class arguments(argparse.Namespace):
	# Model arguments
	input_dir = './data1'
	output_dir = './output' 
	output_video = '04_0_BogMue'
	data_dir = './data1/04_0_BogMue.data'
	#task 1 3
	reference  = [457, 507]

	#task 2 4
	reference1  = [457, 507]
	reference2  = [50, 100]
	
	target = [0, 50]
	ingore_confidence = True

arg = arguments
