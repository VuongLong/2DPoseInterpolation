# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def process_hub5(method = 1, joint = True):
	interpolation_13_v7(np.copy(Tracking3D))


if __name__ == '__main__':

	data_link = "./checking.txt"
		# Tracking3D, restore  = read_tracking_data3D(arg.data_dir3D)
	Tracking3D, _  = read_tracking_data3D_nan(data_link)
	Tracking3D = Tracking3D.astype(float)
	r3, r4 = process_hub5(method = 5, joint = True)
	print(r3, r4)