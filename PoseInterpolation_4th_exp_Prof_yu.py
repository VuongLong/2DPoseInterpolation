# implemention corresponds formula in section Yu - 04th 07 2019
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


def generate_missing_joint(n, m, number_missing):
	m = m // 3
	l = [x for x in range(n*m)]
	missing_joint = random.sample(l, number_missing)
	matrix = np.random.rand(n,m*3)
	for position in missing_joint:
		xx = position // m
		yy = position % m
		matrix[xx][yy*3] = 0
		matrix[xx][yy*3+1] = 0
		matrix[xx][yy*3+2] = 0
	return matrix

def generate_missing_row(n, m):
	matrix = np.random.rand(n,m)
	arr = [70,71,72]
	for index in arr:
		for x in range(n):
			matrix[x, index] = 0
	return matrix


def generate_missing_col(n, m):
	matrix = np.random.rand(n,m)
	arr = [71,72,73,74,75]
	for index in arr:
		for x in range(m):
			matrix[index, x] = 0
	return matrix

def process_hub5(method = 1, joint = True):
	type_plot = "joint"

	A_N = np.array([])
	for x in arg.reference_task4_3D:
		tmp = np.copy(Tracking3D[x[0]:x[1]])
		if A_N.shape[0] != 0:
			A_N = np.concatenate((A_N, tmp), axis = 1)
		else:
			A_N = np.copy(tmp)
	A_N3 = np.copy(Tracking3D[0: 0+arg.AN_length_3D])
	#A = np.copy(Tracking3D[arg.reference[0]:arg.reference[0]+arg.length3D])
	resultA3 = []
	resultA4 = []
	tmpA3 = []
	tmpA4 = []
	for times in range(20):
		print("current: ", times)
		
		A1 = np.copy(Tracking3D[401:501])
		missing_matrix = generate_missing_row(A1.shape[0], A1.shape[1])
		A1zero = np.copy(A1)
		A1zero[np.where(missing_matrix == 0)] = missing_matrix[np.where(missing_matrix == 0)]

		
		A1_star3 = interpolation_13_v5(np.copy(A_N3),np.copy(A1zero), Tracking3D)
		tmpA3.append(np.around(calculate_mse_matrix_Yu(A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)]), decimals = 17))
		tmpA4.append(np.around(calculate_mse_matrix(A1[np.where(A1zero == 0)]- A1_star3[np.where(A1zero == 0)]), decimals = 17))
		# tmpA3.append(np.around(calculate_mse(A1, A1_star3), decimals = 17))
		
		# compute 2nd method
		# A1_star4 = interpolation_24_v5(np.copy(A_N),np.copy(A1zero), Tracking3D)
		# tmpA4.append(np.around(calculate_mse_matrix_Yu(A1[np.where(A1zero == 0)]- A1_star4[np.where(A1zero == 0)]), decimals = 17))
		# tmpA4.append(np.around(calculate_mse(A1, A1_star4), decimals = 17))
	# Tresult = np.asarray(tmpA3).mean()
	Fresult = np.asarray(tmpA3).mean()
	Tresult = np.asarray(tmpA4).mean()
	return Fresult, Tresult



if __name__ == '__main__':
	data_link = ["./data3D/test.data"]
	# data_link = ["./data3D/85_12.txt", "./data3D/HDM_mm_02-02_02_120.txt", "./data3D/HDM_mm_01-02_03_120.txt", "./data3D/HDM_mm_03-02_01_120.txt"]
	Tracking3D, restore  = read_tracking_data3D_v2(data_link[0])
	Tracking3D = Tracking3D.astype(float)
	r3, r4 = predicted_matrix = process_hub5(method = 5, joint = True)
	print(r3, r4)