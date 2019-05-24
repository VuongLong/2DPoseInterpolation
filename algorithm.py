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
	A0 = np.copy(AA)
	A0[np.where(A1 == 0)] = 0
	A_MeanVec = np.mean(A, 0)
	A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
	A_new = A - A_MeanMat
	
	A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
	#colMean = a.sum(0) / (a != 0).sum(0)
	#rowMean = a.sum(1) / (a != 0).sum(1)  
	A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
	A1_new = A1 - A1_MeanMat
	A1_new[np.where(A1 == 0)] = 0
	
	A0_MeanVec = A0.sum(0) / (A0 != 0).sum(0)
	A0_MeanMat = np.tile(A0_MeanVec,(A0.shape[0], 1))
	A0_new = A0 - A0_MeanMat
	A0_new[np.where(A1 == 0)] = 0

	return A_new.T, A0_new.T, A1_new.T, A1_MeanMat.T, A0_MeanMat.T


def deficiency_matrix3(AA, AA0, AA1):
	A = np.copy(AA)
	A1 = np.copy(AA1)
	A_MeanVec = np.mean(A, 0)
	A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
	A_new = A - A_MeanMat
	
	A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
	# print("check", (A1 != 0).sum(0))
	#colMean = a.sum(0) / (a != 0).sum(0)
	#rowMean = a.sum(1) / (a != 0).sum(1)  
	A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
	A1_new = A1 - A1_MeanMat
	A1_new[np.where(A1 == 0)] = 0
	
	A0 = np.copy(AA0)
	A0_MeanVec = np.mean(A0, 0)
	A0_MeanMat = np.tile(A0_MeanVec,(A0.shape[0], 1))
	A0_new = A0 - A0_MeanMat
	A0_new[np.where(A1 == 0)] = 0
	return A_new.T, A0_new.T, A1_new.T, A1_MeanMat.T, A0_MeanMat.T


def interpolation_13(AA, AA1):
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix(AA, AA1)

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

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	A1[np.where(A1 == 0)] = A1star[np.where(A1 == 0)]
	A0[np.where(A0 == 0)] = A0star[np.where(A0 == 0)]

	return A1.T, A0.T



def interpolation_3(AA, AA0, AA1):
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix3(AA, AA0, AA1)

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

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	A1[np.where(A1 == 0)] = A1star[np.where(A1 == 0)]
	A0[np.where(A0 == 0)] = A0star[np.where(A0 == 0)]

	return A1.T, A0.T


def interpolation_24(AA, AA1):
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix(AA, AA1)

	V = mysvd(np.matmul(A.T, A)) 
	V0 = mysvd(np.matmul(A0.T, A0)) 
	F = np.matmul(V0.T, V)
		
	V1 = mysvd(np.matmul(A1.T, A1))

	A1V1F = np.matmul(np.matmul(A1, V1), F)
	A0V0F = np.matmul(np.matmul(A0, V0), F)
	A1star =  np.matmul(A1V1F, V.T)
	A0star =  np.matmul(A0V0F, V.T)

	A1star = A1star + A1_MeanMat
	A0star = A0star + A0_MeanMat

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	A1[np.where(A1 == 0)] = A1star[np.where(A1 == 0)]
	A0[np.where(A0 == 0)] = A0star[np.where(A0 == 0)]
	#return A1.T, VTI, A1V1F.reshape(joint_length*frame_length, 1)
	return A1.T, A0.T

def interpolation_4(AA, AA0, AA1):
	A, A0, A1, A1_MeanMat, A0_MeanMat = deficiency_matrix3(AA, AA0, AA1)
	print(A.shape)
	print(A0.shape)
	print(A1.shape)
	V = mysvd(np.matmul(A.T, A)) 
	V0 = mysvd(np.matmul(A0.T, A0)) 
	print(V.shape)
	print(V0.shape)
	F = np.matmul(V0.T, V)
		
	V1 = mysvd(np.matmul(A1.T, A1))

	A1V1F = np.matmul(np.matmul(A1, V1), F)
	A0V0F = np.matmul(np.matmul(A0, V0), F)
	A1star =  np.matmul(A1V1F, V.T)
	A0star =  np.matmul(A0V0F, V.T)

	A1star = A1star + A1_MeanMat
	A0star = A0star + A0_MeanMat

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	A1[np.where(A1 == 0)] = A1star[np.where(A1 == 0)]
	A0[np.where(A0 == 0)] = A0star[np.where(A0 == 0)]
	#return A1.T, VTI, A1V1F.reshape(joint_length*frame_length, 1)
	return A1.T, A0.T

def calculate_mse(X, Y):
	mse = (np.square(X - Y)).mean()
	#print(X)
	#print("check")
	#print(Y)
	return mse


def get_random_joint(A, length, num_joint_missing):
	number_frame_missing = 5
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