import argparse

class arguments(argparse.Namespace):

	length3D = 200
	reference_task4_3D = [[450, 650]]
	reference_task4_3D_source = [[50, 250], [250, 450]]

	ingore_confidence = True

arg = arguments
