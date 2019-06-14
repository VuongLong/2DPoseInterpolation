import numpy as np
import random


def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	#print(Sigma) # Sigma is a row vector
	#Sigma_mat = np.mat(np.eye(75) * Sigma[:75]) # change Sigma to a matrix 
	#print(Sigma_mat)
	return U


def deficiency_matrix(AA, AA0, AA1, shift):
	A = np.copy(AA)
	A1 = np.copy(AA1)
	A0 = np.copy(AA0)
	A0[np.where(A1 == 0)] = 0

	A_MeanVec = np.mean(A, 0)
	A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
	A_new = np.copy(A - A_MeanMat)


	A0_MeanVec = A0.sum(0) / (A0 != 0).sum(0)
	A0_MeanMat = np.tile(A0_MeanVec,(A0.shape[0], 1))
	A0_new = np.copy(A0 - A0_MeanMat)
	A0_new[np.where(A1 == 0)] = 0
	
	A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
	A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
	A1_new = np.copy(A1 - A1_MeanMat)
	A1_new[np.where(A1 == 0)] = 0
	
	if not shift:
		A1_new = np.copy(A0_new)
		A1_MeanMat = np.copy(A0_MeanMat)


	# A0_new1 = np.copy(A0[0:100, 0:50])
	# A0_MeanMat = np.copy(A_MeanMat[0:100, 0:50]) 
	# A0_new1[np.where(A1 == 0)] = 0	

	# print(A0_new1.shape)
	# print(A0_new.shape)
	return np.copy(A_new.T), np.copy(A0_new.T), np.copy(A1_new.T), np.copy(A1_MeanMat.T), np.copy(A0_MeanMat.T)


def interpolation_13(AA, AA0, AA1, shift):
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix(AA, AA0, AA1, shift)

	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T)) 
	TMat = np.matmul(U0.T, U)  #U = U0TMat
	U1 = mysvd(np.matmul(A1, A1.T)) 
	


	TTU1TA1 = np.matmul(TMat.T, np.matmul(U1.T, A1))
	TTU0TA0 = np.matmul(TMat.T, np.matmul(U0.T, A0))
	A1star =  np.matmul(np.matmul(np.matmul(U, TMat.T), U1.T), A1)
	A0star =  np.matmul(np.matmul(np.matmul(U, TMat.T), U0.T), A0)
	# A1star =  np.matmul(U, TTU1TA1)
	# A0star = np.matmul(U, TTU0TA0)
	

	A1star = A1star + A1_MeanMat
	A0star = A0star + A0_MeanMat
	
	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	#  for task 5

	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

	# I = np.identity(frame_length)
	I = np.eye(frame_length)
	IUT = np.kron(I, U.T)

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]
	A0[np.where(AA1.T == 0)] = A0star[np.where(AA1.T == 0)]	

	# return A1.T, A0.T, IUT, TTU1TA1.reshape(joint_length*frame_length, 1)
	return A1.T, A0.T, IUT, np.ravel(TTU1TA1, order='F')


def interpolation_24(AA, AA0, AA1, shift):
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix(AA, AA0, AA1, shift)
	
	V = mysvd(np.matmul(A.T, A)) 
	V0 = mysvd(np.matmul(A0.T, A0)) 
	F = np.matmul(V0.T, V)
	V1 = mysvd(np.matmul(A1.T, A1))
	

	A1V1F = np.matmul(np.matmul(A1, V1), F)
	A0V0F = np.matmul(np.matmul(A0, V0), F)
	A1star =  np.matmul(np.matmul(np.matmul(A1, V1), F), V.T)
	A0star =  np.matmul(np.matmul(np.matmul(A0, V0), F), V.T)	
	# A1star =  np.matmul(A1V1F, V.T)
	# A0star =  np.matmul(A0V0F, V.T)

	A1star = A1star + A1_MeanMat
	A0star = A0star + A0_MeanMat
	
	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	# for task 5

	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

	# I = np.identity(joint_length)
	I = np.eye(joint_length)
	VTI = np.kron(V.T, I)
	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]
	A0[np.where(AA1.T == 0)] = A0star[np.where(AA1.T == 0)]
	
	return A1.T, A0.T, VTI, np.ravel(A1V1F, order='F'), A1_MeanMat


def interpolation(A1, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat):
	A_new = np.copy(A1)
	A = np.concatenate((IUT, VTI), axis=0)
	B = np.concatenate((TTU1TA1R, A1V1FR), axis=0)
	X = np.linalg.lstsq(A, B)
	Astar = X[0].reshape(A1.shape[0],A1.shape[1]) + A1_MeanMat.T
	A_new[np.where(A1 == 0)] = Astar[np.where(A1 == 0)]
	return A_new


def calculate_mse(X, Y):
	mse = (np.square(X - Y)).mean()
	mse = (np.sqrt(mse))
	return mse


def get_random_joint(A, length, num_joint_missing):
	number_frame_missing = 15
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	for x in missing_frame_arr:
		for xx in range(num_joint_missing):
			indices = random.randint(0, 24)
			AA[x, indices*2] = 0
			AA[x, indices*2+1] = 0
	return AA


def get_removed_peice(A, length, number_frame_missing):
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	for x in missing_frame_arr:
		for i in range(AA[x].size):
			AA[x][i] = 0
	return AA 