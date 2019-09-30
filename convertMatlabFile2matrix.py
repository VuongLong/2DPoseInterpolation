import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys

if __name__ == '__main__':

	Tracking3D, restore  = read_tracking_data3D_v2("./data3D/BokMue.txt")
	Tracking3D = Tracking3D.astype(float)
	# print(Tracking3D.shape)
	Tracking3D = Tracking3D.T
	# np.savetxt("checkdata.txt", Tracking3D.T, fmt = "%.3f")
	f = open("./data3D/BokMue_take_001_Data.txt", "w")
	for x in range(Tracking3D.shape[0]):
		line = ""
		frame = Tracking3D[x]
		for i in range(len(frame)-1):
			line += str(frame[i]) + ", "
		line += str(frame[-1])
		line += "\n"
		f.write(line)
	print(f.close())