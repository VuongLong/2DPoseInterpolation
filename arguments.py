import argparse

class arguments(argparse.Namespace):

	length3D = 400
	reference_task4_3D = [[1050, 1450]]
	reference_task4_3D_source = [[250, 650], [650, 1050], [1450, 1850], [1850, 2250]]
# [[350, 450], [150, 250], [550, 650]]
	ingore_confidence = True

arg = arguments
