# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


if __name__ == '__main__':

	data_link1 = "./PCA.txt"

	Data1, _  = read_tracking_data3D_v3(data_link1)
	Data1 = Data1.astype(float)
	
	data_link2 = "./interpolate.txt"
	# Data2, _  = read_tracking_data3D_nan(data_link2)
	Data2, _  = read_tracking_data3D_v3(data_link2)
	Data2 = Data2.astype(float)
	print(np.sum(np.abs(Data1-Data2)))
	np.savetxt("result_compare.txt",Data1 - Data2, fmt = "%.5f")

