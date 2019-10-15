# implemention corresponds 1st formula to check result
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


if __name__ == '__main__':
	color = ['g', 'r', 'y', 'b', 'k' , 'c']

	data_link = ["./data3D/135_02.txt"]
	Tracking3D, _  = read_tracking_data3D_v2(data_link[0])
	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D.T
	number_joint = Tracking3D.shape[0] // 3
	title = "Joint curve"
	fig = plt.figure()
	fig.suptitle(title, fontsize=16)
	for joint_index in range(number_joint):
		x1 = joint_index*3
		x2 = joint_index*3+1
		x3 = joint_index*3+2
		xx = Tracking3D[x1]
		yy = Tracking3D[x2]
		zz = Tracking3D[x3]
		
		plt.plot(np.arange(Tracking3D.shape[1]), xx, linewidth=1)


		# ax1 = plt.subplot("311")
		# ax1.set_ylabel(' x axis ')
		# ax1.plot(np.arange(Tracking3D.shape[1]), xx, color='g', linewidth=1)

		# ax2 = plt.subplot("312")
		# ax2.set_ylabel(' y axis ')
		# ax2.plot(np.arange(Tracking3D.shape[1]), yy, color='r', linewidth=1)
		
		# ax3 = plt.subplot("313")
		# ax3.set_ylabel(' z axis ')
		# ax3.plot(np.arange(Tracking3D.shape[1]), zz, color='y', linewidth=1)
		# ax3.set_xlabel('Frames')

	fig.set_size_inches(6, 8)
	plt.savefig('./output/'+title+'.png', dpi=600)
