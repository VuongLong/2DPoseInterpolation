import argparse

class arguments(argparse.Namespace):

	length3D = 100
	reference_task4_3D = [[450, 550]]
	reference_task4_3D_source = [[50, 150], [150, 250], [350, 450], [50, 150], [150, 250], [550, 650]]
# [[350, 450], [150, 250], [550, 650]]
	ingore_confidence = True

arg = arguments
