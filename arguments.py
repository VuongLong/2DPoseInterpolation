import argparse

class arguments(argparse.Namespace):

	length3D = 100
	reference_task4_3D = [[450, 550]]
	reference_task4_3D_source = [[150, 250], [250, 350], [350, 450], [550, 650]]

	ingore_confidence = True

arg = arguments
