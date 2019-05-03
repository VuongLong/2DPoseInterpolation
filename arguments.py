import argparse

class arguments(argparse.Namespace):
	# Model arguments
	input_dir = './data1'
	output_dir = './output' 
	output_video = '04_0_BogMue'
	data_dir = './data1/04_0_BogMue.data'
	reference  = [0, 432]
	target = [0, 432]
	ingore_confidence = True

arg = arguments
