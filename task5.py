import numpy as np
import random


def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	#print(Sigma) # Sigma is a row vector
	#Sigma_mat = np.mat(np.eye(75) * Sigma[:75]) # change Sigma to a matrix 
	#print(Sigma_mat)
	return U


def deficiency_matrix(AA, AA1):
	A = np.copy(AA)
	A1 = np.copy(AA1)
	A_MeanVec = np.mean(A, 0)
	A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
	A_new = A - A_MeanMat
	
	A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
	#colMean = a.sum(0) / (a != 0).sum(0)
	#rowMean = a.sum(1) / (a != 0).sum(1)  
	A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
	A1_new = A1 - A1_MeanMat
	A1_new[np.where(A1 == 0)] = 0
	
	A0 = A
	A0_new = A0 - A_MeanMat
	A0_new[np.where(A1 == 0)] = 0
	return A_new.T, A0_new.T, A1_new.T, A1_MeanMat.T, A_MeanMat.T


def interpolation_13(AA, AA1):
	A = np.copy(AA)
	A1 = np.copy(AA1)
	print(A.shape)
	print(A1.shape)
	A#0 = np.copy(A)
	#A0[np.where(A1 == 0)] = 0
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix(A, A1)

	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T)) 
	TMat = np.matmul(U0.T, U)  #U = U0TMat
		
	U1 = mysvd(np.matmul(A1, A1.T)) 
	
	TTU1TA1 = np.matmul(TMat.T, np.matmul(U1.T, A1))
	TTU0TA0 = np.matmul(TMat.T, np.matmul(U0.T, A0))
	A1star =  np.matmul(U, TTU1TA1)
	A0star =  np.matmul(U, TTU0TA0)

	A1star = A1star + A1_MeanMat
	A0star = A0star + A0_MeanMat

	print(A1star)
	# for task 5
	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

	I = np.identity(frame_length)
	IUT = np.kron(I, U.T)

	#print(np.where(A1 == 0))
	#print(Astar[np.where(A1 == 0)])

	A1[np.where(A1 == 0)] = A1star[np.where(A1 == 0)]
	A0[np.where(A0 == 0)] = A0star[np.where(A0 == 0)]
	return A1.T, IUT, TTU1TA1.reshape(joint_length*frame_length, 1), A0.T

def interpolation_24(A, A1):
	#A_new, A0_new, A1_new, A1_MeanMat= deficiency_matrix(A, A1)
	A0 = A
	A0[np.where(A1 == 0)] = 0

	V = mysvd(np.matmul(A.T, A)) 
	V0 = mysvd(np.matmul(A0.T, A0)) 
	F = np.matmul(V0.T, V)
		
	V1 = mysvd(np.matmul(A1.T, A1))

	A1V1F = np.matmul(np.matmul(A1, V1), F)
	Astar =  np.matmul(A1V1F, V.T)

	# for task 5
	joint_length = Astar.shape[0]
	frame_length = Astar.shape[1]

	I = np.identity(joint_length)
	VTI = np.kron(V.T, I)

	A1[np.where(A1 == 0)] = Astar[np.where(A1 == 0)]
	return A1.T, VTI, A1V1F.reshape(joint_length*frame_length, 1)

# AX=B
def interpolation(A1, IUT, TTU1TA1R, VTI, A1V1FR):
	A = np.concatenate((IUT, VTI), axis=0)
	B = np.concatenate((TTU1TA1R, A1V1FR), axis=0)
	X = np.linalg.lstsq(A, B)
	A1 = X[0].reshape(A1.shape[0],A1.shape[1])
	return A1.T

'''
def random_drop_joint(A, num_drop=[3, 5]):
	dims = A.shape
	A = A.reshape(dims[0], 25, int(dims[1] / 25))
	for i in range(dims[0]):
		drop = np.random.randint(num_drop[0]) + num_drop[1] - num_drop[0] + 1
		indices = np.random.randint(25, size=drop)
		for idx in indices:
			A[i, idx] = 0
	return A
'''


def calculate_mse(X, Y):
	mse = (np.square(X - Y)).mean()
	print(X)
	print("check")
	print(Y)
	return mse


def get_random_joint(A, length, num_joint_missing):
	number_frame_missing = 5
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	for x in missing_frame_arr:
		for xx in range(num_joint_missing):
			indices = random.randint(0, 24)
			AA[indices*2, x] = 0
			AA[indices*2+1, x] = 0
	return AA
