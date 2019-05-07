import numpy as np


def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	#print(Sigma) # Sigma is a row vector
	#Sigma_mat = np.mat(np.eye(75) * Sigma[:75]) # change Sigma to a matrix 
	#print(Sigma_mat)
	return U


def deficiency_matrix(A, A1):
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
	return A_new, A0_new, A1_new, A1_MeanMat


def interpolation_13(A, A1):
	A_new, A0_new, A1_new, A1_MeanMat = deficiency_matrix(A, A1)
	
	U = mysvd(np.matmul(A_new, A_new.T))
	U0 = mysvd(np.matmul(A0_new, A0_new.T)) 
	TMat = np.matmul(U0.T, U)  #U = U0TMat
		
	U1 = mysvd(np.matmul(A1_new, A1_new.T)) 
	
	Astar =  np.matmul(np.matmul(np.matmul(U, TMat.T), U1.T), A1_new)
	Astar = Astar + A1_MeanMat
	
	'''replace zero entities'''    
	A1[np.where(A1 == 0)] = Astar[np.where(A1 == 0)]
	return Astar.T


def interpolation_24(A, A1):
	A_new, A0_new, A1_new, A1_MeanMat= deficiency_matrix(A, A1)
	
	V = mysvd(np.matmul(A_new.T, A_new)) 
	V0 = mysvd(np.matmul(A0_new.T, A0_new)) 
	TMat = np.matmul(V0.T, V)  #U = U0TMat
		
	V1 = mysvd(np.matmul(A1_new.T, A1_new)) 
	
	Astar =  np.matmul(np.matmul(np.matmul(A1_new, V1), TMat), V.T)
	Astar = Astar + A1_MeanMat
	
	'''replace zero entities'''    
	A1[np.where(A1 == 0)] = Astar[np.where(A1 == 0)]
	return Astar.T


def random_drop_joint(A, num_drop=[3, 5]):
	dims = A.shape
	A = A.reshape(dims[0], 25, int(dims[1] / 25))
	for i in range(dims[0]):
		drop = np.random.randint(num_drop[0]) + num_drop[1] - num_drop[0] + 1
		indices = np.random.randint(25, size=drop)
		for idx in indices:
			A[i, idx] = 0
	return A






