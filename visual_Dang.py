import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def read_data(link):
	matrix = []
	f=open(link, 'r')
	for line in f:
		elements = line.split(', ')
		matrix.append(list(map(float, elements)))
	f.close()

	matrix = np.array(matrix) 
	matrix = np.squeeze(matrix)
	print(matrix.shape)
	np.savetxt("checkData.txt", matrix.T, fmt = "%.2f")
	return matrix


if __name__ == '__main__':
	# 135 CMU
	# dad_arr = [[0, 2], [1, 3], [2, 4], [3, 4], [9, 10], [10, 11], [11, 12], [12, 14], [14, 13], [14, 15], 
	# [16, 17], [17, 18], [18, 19], [19, 21], [21, 20], [21, 22],
	# [23, 27], [27, 28], [28, 29], [29, 30], [30, 31], [30, 32], [30, 33], 
	# [24, 34], [34, 35], [35, 36], [36, 37], [37, 38], [37, 39], [37, 40],
	# [4, 5],  [9, 4], [16, 4], [25, 23], [24, 26], [5, 25], [5, 26],[7, 23], [7, 24], [6, 16], [6, 9], [6, 7]]
	# [7, 23], [7, 24, [6, 16], [6, 9], [6, 7]]
	# Bvh
	dad_arr = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [1, 6], [6, 7], [7, 8], [8, 9], [1, 10], 
	[10, 11], [11, 12], [12, 13], [13, 14], [10, 15], [15, 16], [16, 17], [17, 18]]
	# 0 head
	# 1 neck
	# 2 LShoulder
	# 3 LElbow
	# 4 LWrist
	# 5
	# 6 RShoulder
	# 7 RElbow
	# 8 RWrist
	# 9
	# 10 Midhip
	# 11 LHip 
	# 12 LKnee
	# 13 LAnkle
	# 14 
	# 15 RHip
	# 16 RKnee
	# 17 RAnkle
	# 18
	Tracking3D  = read_data("./Data3D/ChaiMue_take_001_Data.txt")
	# Tracking3D  = read_data("./result.txt")
	Tracking3D = Tracking3D.astype(float)
	# np.savetxt("ChaiMue_take_001_Data.txt", Tracking3D, fmt = "%.3f", delimiter = ", ")
	fig = plt.figure()
	fig.set_size_inches(8, 12)
	ax = fig.add_subplot(111, projection='3d')
	for index in range(0, 1):
		plt.cla()
		print(index)
		frame = Tracking3D[index]
		n_joints = len(frame) // 3
		xs = []
		ys = []
		zs = []
		for x in range(n_joints):
			xs.append(frame[x*3])
			ys.append(frame[x*3+1])
			zs.append(frame[x*3+2])
		# print(xs)
		# print(ys)
		# print(zs)
		# ax.plot(xs, ys, zs, 'r.')
		# tmp = 1
		# for x in range(len(dad)):
		# 	xxs = [xs[tmp],xs[tmp+1]]
		# 	yys = [ys[tmp],ys[tmp+1]]
		# 	zzs = [zs[tmp],zs[tmp+1]]
		# 	ax.plot(xxs, yys, zzs, 'b')

		for x in range(len(dad_arr)):
			dad = dad_arr[x][0]
			child = dad_arr[x][1]
			my_color = 'b'
			if x == 10:
				my_color = 'g'
			xxs = [xs[dad],xs[child]]
			yys = [ys[dad],ys[child]]
			zzs = [zs[dad],zs[child]]
			ax.plot(xxs, yys, zzs, my_color)

		ax.plot(xs, ys, zs, 'r.')
		# CMU
		# ax.set_xlim3d(-1500, 1500)
		# ax.set_ylim3d(-1500, 1500)
		# ax.set_zlim3d(0, 1500)

		# BVH
		ax.set_xlim3d(-70, 70)
		ax.set_ylim3d(-50, 300)
		ax.set_zlim3d(0, 300)


		# ax.set_xlabel('X Label')
		# ax.set_ylabel('Y Label')
		# ax.set_zlabel('Z Label')
		plt.pause(0.0001)

	plt.show()
	plt.close()

