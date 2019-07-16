import numpy as np
import random
from arguments import arg

def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	#print(Sigma) # Sigma is a row vector
	#Sigma_mat = np.mat(np.eye(75) * Sigma[:75]) # change Sigma to a matrix
	#print(Sigma_mat)
	return U


def deficiency_matrix(AA, AA0, AA1, shift, option = None):
	A = np.copy(AA)
	A1 = np.copy(AA1)
	A0 = np.copy(AA0)
	AAA = np.copy(AA0)
	if option == None:
		A_MeanVec = np.mean(A, 0)
		A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
		A_new = np.copy(A - A_MeanMat)

		AAA_MeanVec = np.mean(AAA, 0)
		AAA_MeanMat = np.tile(AAA_MeanVec, (AAA.shape[0], 1))
		AAA_new = np.copy(AAA - AAA_MeanMat)

		A0_MeanVec = A0.sum(0) / (A0 != 0).sum(0)
		A0_MeanMat = np.tile(A0_MeanVec,(A0.shape[0], 1))
		A0_new = np.copy(A0 - A0_MeanMat)
		A0_new[np.where(A1 == 0)] = 0
	else:
		A_MeanMat = option[0]
		A_new = np.copy(A - A_MeanMat)

		A0_MeanMat = option[1]
		A0_new = np.copy(A0 - A0_MeanMat)
		AAA_new = np.copy(A0_new)
		A0_new[np.where(A1 == 0)] = 0

	A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
	A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
	A1_new = np.copy(A1 - A1_MeanMat)
	A1_new[np.where(A1 == 0)] = 0

	if not shift:
		A1_new = np.copy(A0_new)
		A1_MeanMat = np.copy(A0_MeanMat)

	return np.copy(A_new.T), np.copy(A0_new.T), np.copy(A1_new.T), np.copy(A1_MeanMat.T), np.copy(A0_MeanMat.T), np.copy(AAA_new.T)


def get_Tmatrix13(AA, AA1):
	K = arg.AN_length / arg.length
	# change AN_length as well as length to ""+3D when run 3D experiments
	U = mysvd(np.matmul(AA, AA.T))
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = 50*i+0
		r = 50*i+50
		tmp = np.copy(AA[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		list_U0.append(mysvd(np.matmul(list_A0[i], list_A0[i].T)))
	UTA = np.hstack([np.matmul(U.T, list_A[i]) for i in range(K)])
	A0TU0 = np.vstack([np.matmul(list_A0[i].T, list_U0[i]) for i in range(K)])

	U0TA0 = np.hstack([np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)])
	right_hand = np.matmul(U0TA0, U0TA0.T)
	right_hand_inv = np.linalg.inv(right_hand)
	Tmatrix = np.matmul(np.matmul(UTA, A0TU0), right_hand_inv)
	return Tmatrix.T


# another formula T
def get_Tmatrix13_v2(AA, AA1):
	K = arg.AN_length / arg.length
	# change AN_length as well as length to ""+3D when run 3D experiments
	U = mysvd(np.matmul(AA, AA.T))
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = arg.length*i+0
		r = arg.length*i+arg.length
		tmp = np.copy(AA[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		list_U0.append(mysvd(np.matmul(list_A0[i], list_A0[i].T)))
	UTA = np.hstack([np.matmul(U.T, list_A[i]) for i in range(K)])
	A0TU0 = np.vstack([np.matmul(list_A0[i].T, list_U0[i]) for i in range(K)])

	U0TA0 = np.hstack([np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)])
	right_hand = np.matmul(U0TA0, A0TU0)
	right_hand_inv = np.linalg.inv(right_hand)
	Tmatrix = np.matmul(np.matmul(UTA, A0TU0), right_hand_inv)
	return Tmatrix.T

def get_zero(matrix):
	counter = 0
	for x in matrix:
		if x >= 1: counter += 1
	return matrix.shape[0] - counter

# latest T formula
def get_Tmatrix13_v3(AA, AA1):
	K = arg.AN_length / arg.length
	# change AN_length as well as length to ""+3D when run 3D experiments
	U = mysvd(np.matmul(AA, AA.T))
	_, Sigma, _ = np.linalg.svd(np.matmul(AA, AA.T))
	ksmall = 0
	list_A = []
	list_A0 = []
	list_U0 = []
	list_U0new = []
	for i in range(K):
		l = arg.length*i+0
		r = arg.length*i+arg.length
		tmp = np.copy(AA[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		U0tmp, Sigma, _ = np.linalg.svd(np.matmul(list_A0[i], list_A0[i].T))
		list_U0.append(U0tmp)
		ksmall = max(ksmall, get_zero(Sigma))

	U_new = np.copy(U[:,:ksmall])
	for i in range(K):
		list_U0new.append(list_U0[i][:,:ksmall])

	UTA = np.hstack([np.matmul(U_new.T, list_A[i]) for i in range(K)])
	A0TU0 = np.vstack([np.matmul(list_A0[i].T, list_U0new[i]) for i in range(K)])

	U0TA0 = np.hstack([np.matmul(list_U0new[i].T, list_A0[i]) for i in range(K)])
	right_hand = np.matmul(U0TA0, A0TU0)
	right_hand_inv = np.linalg.inv(right_hand)
	Tmatrix = np.matmul(np.matmul(UTA, A0TU0), right_hand_inv)
	return U_new, Tmatrix.T, ksmall



def get_matmul6(m1, m2, m3, m4, m5, m6):
	return np.matmul(m1, np.matmul(m2, np.matmul(m3, np.matmul(m4, np.matmul(m5,m6)))))


def get_Tmatrix24(AA, AA1):
	K = arg.AN_length / arg.length
	# change AN_length as well as length to ""+3D when run 3D experiments
	V = mysvd(np.matmul(AA.T, AA))
	list_A = []
	list_A0 = []
	list_V0 = []
	for i in range(K):
		l = 50*i+0
		r = 50*i+50
		tmp = np.copy(AA[l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		list_V0.append(mysvd(np.matmul(list_A0[i].T, list_A0[i])))
	AV = np.vstack([np.matmul(list_A[i], V) for i in range(K)])
	V0TA0T = np.hstack([np.matmul(list_V0[i].T, list_A0[i].T) for i in range(K)])

	A0V0 = np.vstack([np.matmul(list_A0[i], list_V0[i]) for i in range(K)])
	left_hand = np.matmul(A0V0.T, A0V0)
	left_hand_inv = np.linalg.inv(left_hand)
	Tmatrix = np.matmul(np.matmul(left_hand_inv, V0TA0T), AV)
	return Tmatrix


def get_Tmatrix24_v2(AA, AA1):
	K = arg.AN_length / arg.length
	# change AN_length as well as length to ""+3D when run 3D experiments
	V = mysvd(np.matmul(AA.T, AA))
	ksmall = 0
	list_A = []
	list_A0 = []
	list_V0 = []
	list_V0new = []
	for i in range(K):
		l = AA1.shape[0]*i+0
		r = AA1.shape[0]*i+AA1.shape[0]
		tmp = np.copy(AA[l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		U0tmp, Sigma, _ = np.linalg.svd(np.matmul(list_A0[i].T, list_A0[i]))
		list_V0.append(U0tmp)
		ksmall = max(ksmall, get_zero(Sigma))

	V_new = np.copy(V[:,:ksmall])
	for i in range(K):
		list_V0new.append(list_V0[i][:,:ksmall])

	AV = np.vstack([np.matmul(list_A[i], V_new) for i in range(K)])
	V0TA0T = np.hstack([np.matmul(list_V0new[i].T, list_A0[i].T) for i in range(K)])

	A0V0 = np.vstack([np.matmul(list_A0[i], list_V0new[i]) for i in range(K)])
	left_hand = np.matmul(V0TA0T, A0V0)
	left_hand_inv = np.linalg.inv(left_hand)
	Tmatrix = np.matmul(np.matmul(left_hand_inv, V0TA0T), AV)
	return V_new, Tmatrix, ksmall

def reconstruct_interpolate(AA1, Astar, A_MeanMat):
	if np.absolute(sum(Astar[np.where(AA1.T == 0)])) <= 0.01:
		return Astar
	return Astar + A_MeanMat


def interpolation_13(AA, AA0, AA1, shift, option = None, Tmatrix = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)


	# in case of I forget the meaning of AAA
	# AAA is normalized matrix of A0. this matrix will be used as label of A1

	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T))
	U1 = mysvd(np.matmul(A1, A1.T))

	if Tmatrix == None:
		UTA = np.matmul(U.T, AAA)
		U1TA1_T = np.matmul(U1.T, A1).T
		X = np.linalg.lstsq(U1TA1_T, UTA.T)
		TMat1 = X[0]
	else:
		TMat1 = get_Tmatrix13_v2(A, A1)

	TTU1TA1 = np.matmul(TMat1.T, np.matmul(U1.T, A1))
	TTU0TA0 = np.matmul(TMat1.T, np.matmul(U0.T, A0))
	A1star =  np.matmul(np.matmul(np.matmul(U, TMat1.T), U1.T), A1)
	A0star =  np.matmul(np.matmul(np.matmul(U, TMat1.T), U0.T), A0)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)
	A0star = reconstruct_interpolate(AA1, A0star, A0_MeanMat)

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	#  for task 5

	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

	I = np.eye(frame_length)
	IUT = np.kron(I, U.T)

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]
	A0[np.where(AA1.T == 0)] = A0star[np.where(AA1.T == 0)]

	return A1.T, A0.T, IUT, np.ravel(TTU1TA1, order='F')

def interpolation_13_v2(AA, AA0, AA1, shift, option = None, Tmatrix = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)


	# in case of I forget the meaning of AAA
	# AAA is normalized matrix of A0. this matrix will be used as label of A1

	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T))
	U1 = mysvd(np.matmul(A1, A1.T))


	U_new, TMat1, ksmall = get_Tmatrix13_v3(A, A1)
	U1_new = np.copy(U1[:, :ksmall])
	U0_new = np.copy(U0[:, :ksmall])

	TTU1TA1 = np.matmul(TMat1.T, np.matmul(U1_new.T, A1))
	TTU0TA0 = np.matmul(TMat1.T, np.matmul(U0_new.T, A0))
	A1star =  np.matmul(np.matmul(np.matmul(U_new, TMat1.T), U1_new.T), A1)
	A0star =  np.matmul(np.matmul(np.matmul(U_new, TMat1.T), U0_new.T), A0)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)
	A0star = reconstruct_interpolate(AA1, A0star, A0_MeanMat)

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	#  for task 5

	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

	I = np.eye(frame_length)
	IUT = np.kron(I, U.T)

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]
	A0[np.where(AA1.T == 0)] = A0star[np.where(AA1.T == 0)]

	return A1.T, A0.T, IUT, np.ravel(TTU1TA1, order='F')

# this function according to section Yu 8th July 2019
def interpolation_13_v3(AA, AA0, AA1, shift, option = None, Tmatrix = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)


	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T))
	U1 = mysvd(np.matmul(A1, A1.T))

	# in case of I forget the meaning of AAA
	# AAA is normalized matrix of A0. this matrix will be used as label of A1

	K = arg.AN_length / arg.length
	# change AN_length as well as length to ""+3D when run 3D experiments
	ksmall = 0
	list_A = []
	list_A0 = []
	list_U = []
	list_Unew = []
	for i in range(K):
		l = arg.length*i+0
		r = arg.length*i+arg.length
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(A1 == 0)] = A1[np.where(A1 == 0)]
		list_A0.append(np.copy(tmp))
		Utmp, Sigma, _ = np.linalg.svd(np.matmul(list_A[i], list_A[i].T))
		list_U.append(Utmp)
		ksmall = max(ksmall, get_zero(Sigma))

	for i in range(K):
		list_Unew.append(list_U[i][:,:ksmall])


	list_qi = []
	list_left = []
	for counter in range(K):
		list_pi = []
		Qi = np.zeros((arg.length, arg.length))
		for i in range(K):
			Pi = np.zeros((arg.length, arg.length))
			for j in range(K):
				Pi += get_matmul6(list_A0[j].T, list_Unew[counter], list_Unew[counter].T, list_Unew[i], list_Unew[i].T, list_A0[j]) 
			Pi = Pi.reshape(arg.length * arg.length, 1)
			list_pi.append(Pi)
			Qi += np.matmul(np.matmul(np.matmul(list_Unew[counter], list_Unew[counter].T), list_A0[i]).T, list_A[i])
		
		Qi = Qi.reshape(arg.length * arg.length,1)
		list_qi.append(Qi)
		ls_left = np.hstack([list_pi[i] for i in range(K)])
		list_left.append(ls_left)

	left_hand = np.vstack([x for x in list_left])
	right_hand = np.vstack([x for x in list_qi])
	T_list = np.linalg.lstsq(left_hand, right_hand)[0]

	A1star = np.zeros(A1.shape)
	for i in range(K):
		A1star += T_list[i] * np.matmul(np.matmul(list_Unew[i], list_Unew[i].T), A1)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)

	A1 = A1 + A1_MeanMat

	#  for task 5

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]

	return A1.T




def interpolation_24(AA, AA0, AA1, shift, option = None, Tmatrix = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)

	V = mysvd(np.matmul(A.T, A))
	V0 = mysvd(np.matmul(A0.T, A0))
	V1 = mysvd(np.matmul(A1.T, A1))

	if Tmatrix == None:
		AV = np.matmul(AAA, V)
		A1V1 = np.matmul(A1, V1)
		X = np.linalg.lstsq(A1V1, AV)
		F1 = X[0]
	else:
		F1 = get_Tmatrix24(A, A1)

	A1V1F = np.matmul(np.matmul(A1, V1), F1)
	A0V0F = np.matmul(np.matmul(A0, V0), F1)
	A1star =  np.matmul(np.matmul(np.matmul(A1, V1), F1), V.T)
	A0star =  np.matmul(np.matmul(np.matmul(A0, V0), F1), V.T)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)
	A0star = reconstruct_interpolate(AA1, A0star, A0_MeanMat)

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	# for task 5

	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

	I = np.eye(joint_length)
	VTI = np.kron(V.T, I)
	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]
	A0[np.where(AA1.T == 0)] = A0star[np.where(AA1.T == 0)]

	return A1.T, A0.T, VTI, np.ravel(A1V1F, order='F'), A1_MeanMat


def interpolation_24_v2(AA, AA0, AA1, shift, option = None, Tmatrix = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)

	V = mysvd(np.matmul(A.T, A))
	V0 = mysvd(np.matmul(A0.T, A0))
	V1 = mysvd(np.matmul(A1.T, A1))

	V_new, F1, ksmall = get_Tmatrix24_v2(A, A1)
	V0_new = np.copy(V0[:, :ksmall])
	V1_new = np.copy(V1[:, :ksmall])


	A1V1F = np.matmul(np.matmul(A1, V1_new), F1)
	A0V0F = np.matmul(np.matmul(A0, V0_new), F1)
	A1star =  np.matmul(np.matmul(np.matmul(A1, V1_new), F1), V_new.T)
	A0star =  np.matmul(np.matmul(np.matmul(A0, V0_new), F1), V_new.T)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)
	A0star = reconstruct_interpolate(AA1, A0star, A0_MeanMat)

	A1 = A1 + A1_MeanMat
	A0 = A0 + A0_MeanMat

	# for task 5

	joint_length = A1star.shape[0]
	frame_length = A1star.shape[1]

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
	XX =  X[0].reshape(A1.shape[0],A1.shape[1])
	Astar = reconstruct_interpolate(A1.T, X[0].reshape(A1.shape[0],A1.shape[1]), A1_MeanMat.T)
	A_new[np.where(A1 == 0)] = Astar[np.where(A1 == 0)]
	return A_new


def calculate_mse(X, Y):
	mse = (np.square(X - Y)).mean()
	mse = (np.sqrt(mse))
	return mse


def get_random_joint(A, length, num_joint_missing):
	number_frame_missing = 10
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	ll = [x for x in range(25)]
	missing_joint_arr = random.sample(ll, num_joint_missing)

	for x in missing_frame_arr:
		for xx in missing_joint_arr:
			AA[x, xx*2] = 0
			AA[x, xx*2+1] = 0
	return AA


def get_random_joint3D(A, length, num_joint_missing):
	number_frame_missing = 10
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	ll = [x for x in range(15)]
	missing_joint_arr = random.sample(ll, num_joint_missing)

	for x in missing_frame_arr:
		for xx in missing_joint_arr:
			AA[x, xx*3] = 0
			AA[x, xx*3+1] = 0
			AA[x, xx*3+2] = 0
	return AA


def get_random_joint_partially(A, length, num_joint_missing, frame_index):
	AA = np.copy(A)
	ll = [x for x in range(25)]
	missing_joint_arr = random.sample(ll, num_joint_missing)
	for xx in missing_joint_arr:
		AA[frame_index, xx*2] = 0
		AA[frame_index, xx*2+1] = 0
	return AA



def get_random_joint_partially3D(A, length, num_joint_missing, frame_index):
	AA = np.copy(A)
	ll = [x for x in range(15)]
	missing_joint_arr = random.sample(ll, num_joint_missing)
	for xx in missing_joint_arr:
		AA[frame_index, xx*3] = 0
		AA[frame_index, xx*3+1] = 0
		AA[frame_index, xx*3+2] = 0
	return AA


def get_removed_peice(A, length, number_frame_missing):
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	for x in missing_frame_arr:
		for i in range(AA[x].size):
			AA[x][i] = 0
	return AA

def get_removed_peice3D(A, length, number_frame_missing):
	AA = np.copy(A)
	l = [x for x in range(length)]
	missing_frame_arr = random.sample(l, number_frame_missing)
	for x in missing_frame_arr:
		for i in range(AA[x].size):
			AA[x][i] = 0
	return AA


def get_remove_row(A, length, num_row_missing):
	number_frame_missing = 15
	AA = np.copy(A)
	arr = random.sample(arg.missing_row_arr, num_row_missing)
	for index in arr:
		for x in range(1, length-1):
			AA[x, index*2] = 0
			AA[x, index*2+1] = 0
	return AA


def get_remove_row3D(A, length, num_row_missing):
	number_frame_missing = 15
	AA = np.copy(A)
	arr = random.sample(arg.missing_row_arr, num_row_missing)
	for index in arr:
		for x in range(1, length-1):
			AA[x, index*3] = 0
			AA[x, index*3+1] = 0
			AA[x, index*3+2] = 0
	return AA
