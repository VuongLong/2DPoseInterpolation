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
	# A = np.copy(AA.T)
	# A1 = np.copy(AA1.T)
	# A0 = np.copy(AA0.T)
	# AAA = np.copy(AA0.T)
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

	
	# return np.copy(A_new), np.copy(A0_new), np.copy(A1_new), np.copy(A1_MeanMat), np.copy(A0_MeanMat), np.copy(AAA_new)
	return np.copy(A_new.T), np.copy(A0_new.T), np.copy(A1_new.T), np.copy(A1_MeanMat.T), np.copy(A0_MeanMat.T), np.copy(AAA_new.T)



def deficiency_matrix2(AA, AA1):
	A = np.copy(AA)
	A1 = np.copy(AA1)

	A_MeanVec = np.mean(A, 0)
	A_MeanMat = np.tile(A_MeanVec, (A.shape[0], 1))
	A_new = np.copy(A - A_MeanMat)

	A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
	A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0], 1))
	A1_new = np.copy(A1 - A1_MeanMat)
	A1_new[np.where(A1 == 0)] = 0
	
	return np.copy(A_new.T), np.copy(A1_new.T), np.copy(A1_MeanMat.T)



def get_Tmatrix13(AA, AA1):
	length_clip = arg.length
	length_sequence = arg.AN_length
	# length_clip = arg.length3D
	# length_sequence = arg.AN_length_3D	

	K = length_sequence / length_clip
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


def get_zero(matrix):
	counter = 0
	for x in matrix:
		if x > 0.01: counter += 1
	return counter

# latest T formula
def get_Tmatrix13_v2(AA, AA1):
	# length_clip = arg.length
	# length_sequence = arg.AN_length
	length_clip = arg.length3D
	length_sequence = AA.shape[1]	

	K = length_sequence // length_clip
	# change AN_length as well as length to ""+3D when run 3D experiments
	U = mysvd(np.matmul(AA, AA.T))
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(AA[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		U0tmp, Sigma, _ = np.linalg.svd(np.matmul(list_A0[i], list_A0[i].T))
		list_U0.append(U0tmp)

	UTA = np.hstack([np.matmul(U.T, list_A[i]) for i in range(K)])
	A0TU0 = np.vstack([np.matmul(list_A0[i].T, list_U0[i]) for i in range(K)])

	U0TA0 = np.hstack([np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)])
	right_hand = np.matmul(U0TA0, A0TU0)
	right_hand_inv = np.linalg.inv(right_hand)
	Tmatrix = np.matmul(np.matmul(UTA, A0TU0), right_hand_inv)
	return U, Tmatrix



def get_matmul6(m1, m2, m3, m4, m5, m6):
	return np.matmul(m1, np.matmul(m2, np.matmul(m3, np.matmul(m4, np.matmul(m5,m6)))))

def get_matmulPi8(m1, m2, m3, m4, m5, m6, m7):
	result = np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(m1, m2),m3),m4.T),m4),m5),m6),m7)
	return result

def get_Tmatrix24(AA, AA1):
	length_clip = arg.length
	length_sequence = arg.AN_length
	# length_clip = arg.length3D
	# length_sequence = arg.AN_length_3D	

	K = length_sequence / length_clip
	# change AN_length as well as length to ""+3D when run 3D experiments
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
	# length_clip = arg.length
	# length_sequence = arg.AN_length
	length_clip = arg.length3D
	length_sequence = arg.AN_length_3D	

	K = length_sequence // length_clip
	# change AN_length as well as length to ""+3D when run 3D experiments
	V = mysvd(np.matmul(AA.T, AA))
	list_A = []
	list_A0 = []
	list_V0 = []
	for i in range(K):
		l = AA1.shape[0]*i+0
		r = AA1.shape[0]*i+AA1.shape[0]
		tmp = np.copy(AA[l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1 == 0)] = AA1[np.where(AA1 == 0)]
		list_A0.append(np.copy(tmp))
		U0tmp, Sigma, _ = np.linalg.svd(np.matmul(list_A0[i].T, list_A0[i]))
		list_V0.append(U0tmp)

	AV = np.vstack([np.matmul(list_A[i], V) for i in range(K)])
	V0TA0T = np.hstack([np.matmul(list_V0[i].T, list_A0[i].T) for i in range(K)])

	A0V0 = np.vstack([np.matmul(list_A0[i], list_V0[i]) for i in range(K)])
	left_hand = np.matmul(V0TA0T, A0V0)
	left_hand_inv = np.linalg.inv(left_hand)
	Tmatrix = np.matmul(np.matmul(left_hand_inv, V0TA0T), AV)
	return V, Tmatrix

def reconstruct_interpolate(AA1, Astar, A_MeanMat):
	if np.absolute(sum(Astar[np.where(AA1.T == 0)])) <= 0.01:
		return Astar
	return Astar + A_MeanMat


def interpolation_13(AA, AA0, AA1, shift = None, option = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)


	# in case of I forget the meaning of AAA
	# AAA is normalized matrix of A0. this matrix will be used as label of A1

	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T))
	U1 = mysvd(np.matmul(A1, A1.T))

	UTA = np.matmul(U.T, AAA)
	U1TA1_T = np.matmul(U1.T, A1).T
	X = np.linalg.lstsq(U1TA1_T, UTA.T, rcond=None)
	TMat1 = X[0]

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

def interpolation_13_v2(AA, AA1, missing_number = 0):
	A, A1, A1_MeanMat = deficiency_matrix2(AA, AA1)
	# print("ok")
	# B = np.copy(AA)
	# B1 = np.copy(AA1)

	# A_MeanVec = np.mean(B, 0)
	# A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	# A_new = np.copy(B - A_MeanMat)

	# A1_MeanVec = np.mean(B[0:80], 0)
	# A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	# A1_new = np.copy(B[0:80] - A1_MeanMat)
	# A1_new[np.where(B1 == 0)] = 0
	
	# A = A_new.T
	# A1 = A1_new.T
	# A1_MeanMat = A1_MeanMat.T
	# in case of I forget the meaning of AAA
	# AAA is normalized matrix of A0. this matrix will be used as label of A1

	U = mysvd(np.matmul(A, A.T))
	U1 = mysvd(np.matmul(A1, A1.T))


	U_new, TMat1 = get_Tmatrix13_v2(A, A1)

	A1star =  np.matmul(np.matmul(np.matmul(U, TMat1), U1.T), A1)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)

	A1 = A1 + A1_MeanMat

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]

	return A1.T

# this function according to section Yu 8th July 2019
def interpolation_13_v3(AA, AA0, AA1, shift, option = None, missing_number = 0):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)

	U = mysvd(np.matmul(A, A.T))
	U0 = mysvd(np.matmul(A0, A0.T))
	U1 = mysvd(np.matmul(A1, A1.T))

	# in case of I forget the meaning of AAA
	# AAA is normalized matrix of A0. this matrix will be used as label of A1

	# length_clip = arg.length
	# length_sequence = arg.AN_length
	length_clip = arg.length3D
	length_sequence = arg.AN_length_3D	

	K = length_sequence // length_clip
	# change AN_length as well as length to ""+3D when run 3D experiments
	ksmall = 1001
	list_A = []
	list_A0 = []
	list_U = []
	list_Unew = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(A1 == 0)] = A1[np.where(A1 == 0)]
		list_A0.append(np.copy(tmp))
		Utmp, Sigma, _ = np.linalg.svd(list_A[i])
		list_U.append(Utmp)
		ksmall = min(ksmall, get_zero(Sigma))
	
	ksmall = min(ksmall, list_U[0].shape[0] - missing_number*3)
	# np.savetxt("Sigma.txt", Sigma, fmt = "%.2f")
	# np.savetxt("A_new1.txt", list_A0[1], fmt = "%.2f")
	# np.savetxt("A_new0.txt", list_U[0], fmt = "%.2f")
	print("1", ksmall)
	for i in range(K):
		list_Unew.append(list_U[i][:,:ksmall])
	
	# print(list_Unew[0].shape)

	# np.savetxt("U1.txt", np.matmul(list_U[1], list_U[1].T)-np.matmul(list_U[0], list_U[0].T), fmt = "%.7f")
	# np.savetxt("U2.txt", np.matmul(list_U[1], list_U[1].T), fmt = "%.2f")
	# print("det:	",np.linalg.det(np.matmul(list_U[1], list_U[1].T)))
	# print("zeros: 	",np.count_nonzero(np.matmul(list_U[1], list_U[1].T)-np.matmul(list_U[0], list_U[0].T)))
	list_qi = []
	list_left = []
	for counter in range(K):
		list_pi = []
		Qi = np.zeros((length_clip, length_clip))
		for i in range(K):
			Pi = np.zeros((length_clip, length_clip))
			for j in range(K):
				Pi += get_matmul6(list_A0[j].T, list_Unew[counter], list_Unew[counter].T, list_Unew[i], list_Unew[i].T, list_A0[j])
			Pi = Pi.reshape(length_clip * length_clip, 1)
			list_pi.append(Pi)
			# np.savetxt("U" +str(i)+".txt", Pi, fmt = "%.2f")
			Qi += np.matmul(np.matmul(np.matmul(list_Unew[counter], list_Unew[counter].T), list_A0[i]).T, list_A[i])

		Qi = Qi.reshape(length_clip * length_clip,1)
		list_qi.append(Qi)
		ls_left = np.hstack([list_pi[i] for i in range(K)])
		list_left.append(ls_left)

	left_hand = np.vstack([x for x in list_left])
	right_hand = np.vstack([x for x in list_qi])
	# np.savetxt("left_hand.txt", left_hand, fmt = "%.2f")
	# np.savetxt("right_hand.txt", right_hand, fmt = "%.2f")
	T_list = np.linalg.lstsq(left_hand, right_hand, rcond=None)[0]
	print(T_list)
	# halt
	A1star = np.zeros(A1.shape)
	for i in range(K):
		A1star += T_list[i] * np.matmul(np.matmul(list_Unew[i], list_Unew[i].T), A1)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)

	A1 = A1 + A1_MeanMat

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]

	return A1.T

def interpolation_13_v4(AA, AA1):
	
	list_mrow = np.where(~AA1.T.any(axis=1)) 
	list_frow = np.where(AA1.T.any(axis=1)) 

	# count the number of patch in reference
	length_clip = AA1.shape[0]
	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[0:length_clip])
	
	# ///////////////////////////////////////////////////////
	selected_patch = np.copy(AA[0:length_clip])
	max_diff = 1000000000000
	mse_arr = []
	tmp_sum = 0
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(AA[l:r])
		tmp_mse = calculate_mse_patch(tmp, AA1)
		mse_arr.append(tmp_mse)
		tmp_sum += tmp_mse
		if tmp_mse < max_diff:
			max_diff = tmp_mse
			selected_patch = tmp

	for x in range(40):
		AA = np.vstack((AA, selected_patch))

	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 

	B = np.copy(AA)
	B1 = selected_patch

	A_MeanVec = np.mean(B, 0)
	A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	A_new = np.copy(B - A_MeanMat)

	A1_MeanVec = np.mean(B1, 0)
	A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	A1_new = np.copy(B1 - A1_MeanMat)
	# np.savetxt("checkA1.txt", A1_MeanMat, fmt = "%.2f")
	# halt

	A1 = np.delete(A1_new.T, list_mrow, 0)
	A = np.copy(A_new.T)
	# axis = 1, get zero row
	
	U = mysvd(np.matmul(A, A.T))

	list_A = []
	list_A0reduce = []
	list_U0reduce = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		tmp_reduce = np.delete(tmp, list_mrow, 0)
		# axis = 0, delete row
		list_A0reduce.append(tmp_reduce)

	U_reduce = np.delete(U, list_mrow, 0)
	U_remain = U[list_mrow]

	for i in range(K):
		list_U0reduce.append(mysvd(np.matmul(list_A0reduce[i], list_A0reduce[i].T)))

	list_alpha0 = [np.matmul(list_U0reduce[i].T, list_A0reduce[i]) for i in range(K)]
	list_alpha = [np.matmul(U.T, list_A[i]) for i in range(K)]

	alpha_alpha0T = np.zeros(np.matmul(list_alpha[0], list_alpha0[0].T).shape)
	alpha0_alpha0T = np.zeros(np.matmul(list_alpha0[0], list_alpha0[0].T).shape)
	for i in range(K):
		alpha_alpha0T += np.matmul(list_alpha[i], list_alpha0[i].T)
		alpha0_alpha0T += np.matmul(list_alpha0[i], list_alpha0[i].T)

	Tmatrix = np.matmul(alpha_alpha0T, np.linalg.inv(alpha0_alpha0T))

	A1reduce = A1
	U1reduce = mysvd(np.matmul(A1reduce, A1reduce.T))
	alpha1 = np.matmul(U1reduce.T, A1reduce)

	A_missing = np.matmul(np.matmul(U_remain, Tmatrix), alpha1)
	print("T:")
	check_interpolation(A_missing)
	tmp = np.zeros(AA1.T.shape)
	tmp[list_frow] = A1
	tmp[list_mrow] = A_missing

	tmp = tmp + A1_MeanMat.T
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	return result.T



def interpolation_13_v5(AA, AA1):
	# np.savetxt("checkmap.txt", AA1, fmt = "%.2f")
	list_mrow = np.where(~AA1.T.any(axis=1)) 
	list_frow = np.where(AA1.T.any(axis=1)) 
	# count the number of patch in reference
	length_clip = AA1.shape[0]
	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[0:length_clip])
	
	# ///////////////////////////////////////////////////////
	selected_patch = np.copy(AA[0:length_clip])
	max_diff = 1000000000000
	mse_arr = []
	tmp_sum = 0
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(AA[l:r])
		tmp_mse = calculate_mse_patch(tmp, AA1)
		mse_arr.append(tmp_mse)
		tmp_sum += tmp_mse
		if tmp_mse < max_diff:
			max_diff = tmp_mse
			selected_patch = tmp
	tmp_mean = tmp_sum / K

	B = np.copy(AA)
	B1 = selected_patch

	A_MeanVec = np.mean(B, 0)
	A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	A_new = np.copy(B - A_MeanMat)

	A1_MeanVec = np.mean(B1, 0)
	A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	A1_new = np.copy(B1 - A1_MeanMat)

	A1 = np.delete(A1_new.T, list_mrow, 0)
	A = np.copy(A_new.T)

	U = mysvd(np.matmul(A, A.T))
	U_reduce = np.delete(U, list_mrow, 0)
	U_remain = U[list_mrow]

	A1reduce = A1

	c1 = np.matmul(U_remain, U_reduce.T)
	c2 = np.linalg.inv(np.matmul(U_reduce, U_reduce.T)) 
	A_missing = np.matmul(np.matmul(c1, c2),A1reduce)
	print("T1:")
	check_interpolation(A_missing)
	tmp = np.zeros(AA1.T.shape)
	tmp[list_frow] = A1
	tmp[list_mrow] = A_missing

	tmp = tmp + A1_MeanMat.T
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = selected_patch.T[np.where(AA1.T == 0)]
	return result.T



def interpolation_T(AA, AA1):
	length_clip = AA1.shape[0]
	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 

	A, A1, A1_MeanMat = deficiency_matrix2(AA, AA1)
	U, UEV, _ = np.linalg.svd(np.matmul(A, A.T))
	
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1.T == 0)] = 0
		list_A0.append(np.copy(tmp))
		U0tmp, U0EV, _ = np.linalg.svd(np.matmul(list_A0[-1], list_A0[-1].T))
		list_U0.append(U0tmp)

	list_alpha0 = [np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)]
	list_alpha = [np.matmul(U.T, list_A[i]) for i in range(K)]

	alpha_alpha0T = np.zeros(np.matmul(list_alpha[0], list_alpha0[0].T).shape)
	alpha0_alpha0T = np.zeros(np.matmul(list_alpha0[0], list_alpha0[0].T).shape)
	for i in range(K):
		alpha_alpha0T += np.matmul(list_alpha[i], list_alpha0[i].T)
		alpha0_alpha0T += np.matmul(list_alpha0[i], list_alpha0[i].T)

	Tmatrix = np.matmul(alpha_alpha0T, np.linalg.inv(alpha0_alpha0T))

	U1 = mysvd(np.matmul(A1, A1.T))
	alpha1 = np.matmul(U1.T, A1)

	A_star = np.matmul(np.matmul(U, Tmatrix), alpha1)
	tmp = A_star + A1_MeanMat
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	print("T0:")
	check_interpolation(tmp[np.where(AA1.T == 0)])
	return result.T


def interpolation_13_v6(AA, AA1):
	# count the number of patch in reference
	length_clip = AA1.shape[0]
	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[0:length_clip])
	
	# # ///////////////////////////////////////////////////////
	# selected_patch = np.copy(AA[0:length_clip])
	# max_diff = 1000000000000
	# mse_arr = []
	# for i in range(K):
	# 	l = length_clip*i+0
	# 	r = length_clip*i+length_clip
	# 	tmp = np.copy(AA[l:r])
	# 	tmp_mse = calculate_mse_patch(tmp, AA1)
	# 	mse_arr.append(tmp_mse)
	# 	if tmp_mse < max_diff:
	# 		max_diff = tmp_mse
	# 		selected_patch = tmp

	# B = np.copy(AA)
	# B1 = selected_patch

	# length_sequence = AA.shape[0]
	# K = length_sequence // length_clip 
	# # ///////////////////////////////////////////////////////
	
	# A_MeanVec = np.mean(B, 0)
	# A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	# A_new = np.copy(B - A_MeanMat)

	# A1_MeanVec = np.mean(B1, 0)
	# A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	# A1_new = np.copy(AA1 - A1_MeanMat)
	# A1_new[np.where(AA1 == 0)] = 0
	## np.savetxt("checkA1.txt", A1_MeanMat, fmt = "%.2f")
	## halt

	A_MeanVec = np.mean(AA, 0)
	A_MeanMat = np.tile(A_MeanVec, (AA.shape[0], 1))
	A_new = np.copy(AA - A_MeanMat)

	A1_MeanVec = AA1.sum(0) / (AA1 != 0).sum(0)
	A1_MeanMat = np.tile(A1_MeanVec,(AA1.shape[0], 1))
	A1_new = np.copy(AA1 - A1_MeanMat)
	A1_new[np.where(AA1 == 0)] = 0
	
	A = np.copy(A_new.T)
	A1 = np.copy(A1_new.T)
	A1_MeanMat = np.copy(A1_MeanMat.T)
	# axis = 1, get zero row
	# ksmall = 0 

	U, UEV, _ = np.linalg.svd(np.matmul(A, A.T))
	# matrix_check = np.matmul(A, A.T)
	# np.savetxt("Sigma.txt", UEV, fmt = "%.2f")
	# np.savetxt("checkA.txt", A.T, fmt = "%.2f")
	# np.savetxt("checkA1.txt", matrix_check, fmt = "%.2f")
	# halt
	# ksmall = max(get_zero())
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1.T == 0)] = 0
		list_A0.append(np.copy(tmp))
		U0tmp, U0EV, _ = np.linalg.svd(np.matmul(list_A0[-1], list_A0[-1].T))
		list_U0.append(U0tmp)

	list_alpha0 = [np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)]
	list_alpha = [np.matmul(U.T, list_A[i]) for i in range(K)]

	alpha_alpha0T = np.zeros(np.matmul(list_alpha[0], list_alpha0[0].T).shape)
	alpha0_alpha0T = np.zeros(np.matmul(list_alpha0[0], list_alpha0[0].T).shape)
	for i in range(K):
		alpha_alpha0T += np.matmul(list_alpha[i], list_alpha0[i].T)
		alpha0_alpha0T += np.matmul(list_alpha0[i], list_alpha0[i].T)

	Tmatrix = np.matmul(alpha_alpha0T, np.linalg.inv(alpha0_alpha0T))

	U1 = mysvd(np.matmul(A1, A1.T))
	alpha1 = np.matmul(U1.T, A1)

	A_star = np.matmul(np.matmul(U, Tmatrix), alpha1)
	tmp = A_star + A1_MeanMat
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	print("T0:")
	check_interpolation(tmp[np.where(AA1.T == 0)])
	return result.T

def interpolation_13_v6_v3(AA, AA1):
	# count the number of patch in reference
	length_clip = AA1.shape[0]
	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[0:length_clip])
	
	# ///////////////////////////////////////////////////////
	selected_patch = np.copy(AA[0:length_clip])
	max_diff = 1000000000000
	mse_arr = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(AA[l:r])
		tmp_mse = calculate_mse_patch(tmp, AA1)
		mse_arr.append(tmp_mse)
		if tmp_mse < max_diff:
			max_diff = tmp_mse
			selected_patch = tmp

	B = np.copy(AA)
	B1 = selected_patch

	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 
	# ///////////////////////////////////////////////////////
	
	A_MeanVec = np.mean(B, 0)
	A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	A_new = np.copy(B - A_MeanMat)

	A1_MeanVec = np.mean(B1, 0)
	A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	A1_new = np.copy(AA1 - A1_MeanMat)
	A1_new[np.where(AA1 == 0)] = 0
	# # np.savetxt("checkA1.txt", A1_MeanMat, fmt = "%.2f")
	# # halt

	A = np.copy(A_new.T)
	A1 = np.copy(A1_new.T)
	A1_MeanMat = np.copy(A1_MeanMat.T)
	# axis = 1, get zero row
	k = 1000
	U, UEV, _ = np.linalg.svd(np.matmul(A, A.T))
	k = min(k , get_zero(UEV))
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1.T == 0)] = A1[np.where(AA1.T == 0)]
		list_A0.append(np.copy(tmp))
		U0tmp, U0EV, _ = np.linalg.svd(np.matmul(list_A0[-1], list_A0[-1].T))
		list_U0.append(U0tmp)
		k = min(k, get_zero(U0EV))
	U = U[:,:k]
	for x in range(K):
		list_U0[x] = list_U0[x][:,:k]
	list_alpha0 = [np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)]
	list_alpha = [np.matmul(U.T, list_A[i]) for i in range(K)]

	alpha_alpha0T = np.zeros(np.matmul(list_alpha[0], list_alpha0[0].T).shape)
	alpha0_alpha0T = np.zeros(np.matmul(list_alpha0[0], list_alpha0[0].T).shape)
	for i in range(K):
		alpha_alpha0T += np.matmul(list_alpha[i], list_alpha0[i].T)
		alpha0_alpha0T += np.matmul(list_alpha0[i], list_alpha0[i].T)

	Tmatrix = np.matmul(alpha_alpha0T, np.linalg.inv(alpha0_alpha0T))

	U1 = mysvd(np.matmul(A1, A1.T))[:,:k]
	alpha1 = np.matmul(U1.T, A1)

	A_star = np.matmul(np.matmul(U, Tmatrix), alpha1)

	tmp = A_star + A1_MeanMat
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	print("T0:")
	check_interpolation(tmp[np.where(AA1.T == 0)])
	return result.T

def interpolation_13_v6_v2(AA, AA1):
	
	combine_matrix = np.vstack((AA, AA1))
	weightScale = 200
	MMweight = 0.02
	[frames, columns] = combine_matrix.shape
	columnindex = np.where(combine_matrix == 0)[1]
	frameindex = np.where(combine_matrix == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(combine_matrix, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, combine_matrix.shape[1]//3)
	Data = np.copy(combine_matrix - MeanMat)
	Data[np.where(combine_matrix == 0)] = 0

	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))

	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps

	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	
	A = np.copy(M_zero[:AA.shape[0]])
	A1 = np.copy(M_zero[-AA1.shape[0]:])
	A = np.copy(A.T)
	A1 = np.copy(A1.T)
	# axis = 1, get zero row


	length_clip = AA1.shape[0]
	length_sequence = AA.shape[0]
	K = length_sequence // length_clip 
	k = 10000 
	U, UEV, _ = np.linalg.svd(np.matmul(A, A.T))
	k = min(k , get_zero(UEV))
	list_A = []
	list_A0 = []
	list_U0 = []
	for i in range(K):
		l = length_clip*i+0
		r = length_clip*i+length_clip
		tmp = np.copy(A[:,l:r])
		list_A.append(np.copy(tmp))
		list_A0.append(np.copy(tmp))
		U0tmp, U0EV, _ = np.linalg.svd(np.matmul(list_A0[-1], list_A0[-1].T))
		list_U0.append(U0tmp)
		k = min(k, get_zero(U0EV))
	U = U[:,:k]
	for x in range(K):
		list_U0[x] = list_U0[x][:,:k]
	list_alpha0 = [np.matmul(list_U0[i].T, list_A0[i]) for i in range(K)]
	list_alpha = [np.matmul(U.T, list_A[i]) for i in range(K)]

	alpha_alpha0T = np.zeros(np.matmul(list_alpha[0], list_alpha0[0].T).shape)
	alpha0_alpha0T = np.zeros(np.matmul(list_alpha0[0], list_alpha0[0].T).shape)
	for i in range(K):
		alpha_alpha0T += np.matmul(list_alpha[i], list_alpha0[i].T)
		alpha0_alpha0T += np.matmul(list_alpha0[i], list_alpha0[i].T)

	Tmatrix = np.matmul(alpha_alpha0T, np.linalg.inv(alpha0_alpha0T))

	U1 = mysvd(np.matmul(A1, A1.T))[:,:k]
	alpha1 = np.matmul(U1.T, A1)

	A_star = np.matmul(np.matmul(U, Tmatrix), alpha1)
	reconstructData = np.vstack((A.T, A_star.T))

	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp[-AA1.shape[0]:])

	final_result = np.copy(AA1)
	final_result[np.where(AA1 == 0)] = result[np.where(AA1 == 0)]
	print("T0:")
	check_interpolation(final_result[np.where(AA1 == 0)])
	return final_result


def get_point(Data, frame, joint):
	point = [ Data[frame, joint*3] , Data[frame, joint*3+1] , Data[frame, joint*3+2 ]]
	return point

def PCA_PLOS1(AA, matrix2):

	weightScale = 200
	MMweight = 0.02
	DistalThreshold = 0.5
	MinCumSV = 0.99
	[frames, columns] = AA.shape
	columnindex = np.where(AA == 0)[1]
	frameindex = np.where(AA == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(AA, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
	Data = np.copy(AA - MeanMat)
	Data[np.where(AA == 0)] = 0
	
	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
	
	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
	m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
	m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
	m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
	N_zero = np.multiply(((N_zero-m6) / m5),m33)
	_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
	U_N_nogap = U_N_nogap_VH.T
	_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
	U_N_zero = U_N_zero_VH.T
	ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
	U_N_nogap = U_N_nogap[:, :ksmall]
	U_N_zero = U_N_zero[:, :ksmall]
	T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
	reconstructData = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
	
	# reverse normalization
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp)

	final_result = np.copy(matrix2)
	final_result[np.where(matrix2 == 0)] = result[np.where(matrix2 == 0)]
	print("T0:")
	check_interpolation(final_result[np.where(matrix2 == 0)])
	return final_result


def PCA_PLOS1_Uversion(AA, matrix2):

	weightScale = 200
	MMweight = 0.02
	DistalThreshold = 0.5
	MinCumSV = 0.99
	[frames, columns] = AA.shape
	columnindex = np.where(AA == 0)[1]
	frameindex = np.where(AA == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(AA, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
	Data = np.copy(AA - MeanMat)
	Data[np.where(AA == 0)] = 0
	
	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
	
	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
	m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
	m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
	m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
	N_zero = np.multiply(((N_zero-m6) / m5),m33)
	U_Matrix, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1))
	U1_Matrix, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1))
	print("checking shape")
	print(M_zero.shape)
	print(N_nogap.shape)
	print(N_zero.shape)
	print(U_Matrix.shape)
	print(U1_Matrix.shape)
	ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
	U_Matrix = U_Matrix[:, :ksmall]
	U1_Matrix = U1_Matrix[:, :ksmall]
	T_matrix = np.matmul(U_Matrix.T , U1_Matrix)
	print(U_Matrix.shape)
	print(T_matrix.T.shape)
	print(U1_Matrix.T.shape)
	print(M_zero.shape)
	reconstructData = np.matmul(np.matmul(np.matmul(U_Matrix, T_matrix.T), U1_Matrix.T), M_zero)
	print("end checking shape")
	
	# reverse normalization
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp)

	final_result = np.copy(matrix2)
	final_result[np.where(matrix2 == 0)] = result[np.where(matrix2 == 0)]
	print("T0:")
	check_interpolation(final_result[np.where(matrix2 == 0)])
	return final_result

def interpolation_24(AA, AA0, AA1, shift, option = None, Tmatrix = None):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)

	V = mysvd(np.matmul(A.T, A))
	V0 = mysvd(np.matmul(A0.T, A0))
	V1 = mysvd(np.matmul(A1.T, A1))

	if Tmatrix == None:
		AV = np.matmul(AAA, V)
		A1V1 = np.matmul(A1, V1)
		X = np.linalg.lstsq(A1V1, AV, rcond=None)
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


def interpolation_24_v2(AA, AA1, missing_number = 0):
	A, A1, A1_MeanMat = deficiency_matrix2(AA, AA1)

	V = mysvd(np.matmul(A.T, A))
	V1 = mysvd(np.matmul(A1.T, A1))

	V_new, F1 = get_Tmatrix24_v2(A, A1)

	A1star =  np.matmul(np.matmul(np.matmul(A1, V1), F1), V.T)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)

	A1 = A1 + A1_MeanMat

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]

	return A1.T


# this function according to section Yu 8th July 2019
def interpolation_24_v3(AA, AA0, AA1, shift, option = None, missing_number = 0):
	A, A0, A1, A1_MeanMat, A0_MeanMat, AAA = deficiency_matrix(AA, AA0, AA1, shift, option)

	V = mysvd(np.matmul(A.T, A))
	V0 = mysvd(np.matmul(A0.T, A0))
	V1 = mysvd(np.matmul(A1.T, A1))

	# length_clip = arg.length
	# length_sequence = arg.AN_length
	length_clip = arg.length3D
	length_sequence = arg.AN_length_3D	

	K = length_sequence // length_clip
	# change AN_length as well as length to ""+3D when run 3D experiments
	ksmall = 1001
	list_A = []
	list_A0 = []
	list_V = []
	list_Vnew = []
	for i in range(K):
		l = A1.shape[0]*i+0
		r = A1.shape[0]*i+A1.shape[0]
		tmp = np.copy(A[l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(A1 == 0)] = A1[np.where(A1 == 0)]
		list_A0.append(np.copy(tmp))
		Vtmp, Sigma, _ = np.linalg.svd(list_A[i].T)
		list_V.append(Vtmp)
		ksmall = min(ksmall, get_zero(Sigma))

	ksmall = min(ksmall, list_V[0].shape[0] - missing_number*3)
	print("2",ksmall)
	# halt

	
	for i in range(K):
		list_Vnew.append(list_V[i][:,:ksmall])
	
	list_qi = []
	list_left = []
	for counter in range(K):
		list_pi = []
		Qi = np.zeros((length_clip, length_clip))
		for i in range(K):
			Pi = np.zeros((length_clip, length_clip))
			for j in range(K):
				Pi += get_matmul6(list_Vnew[counter], list_Vnew[counter].T, list_A0[j].T, list_A0[j], list_Vnew[i], list_Vnew[i].T)
			Pi = Pi.reshape(length_clip * length_clip, 1)
			list_pi.append(Pi)
			Qi += np.matmul(np.matmul(np.matmul(list_A0[i], list_Vnew[counter]), list_Vnew[counter].T).T, list_A[i])

		Qi = Qi.reshape(length_clip * length_clip,1)
		list_qi.append(Qi)
		ls_left = np.hstack([list_pi[i] for i in range(K)])
		list_left.append(ls_left)
	left_hand = np.vstack([x for x in list_left])
	right_hand = np.vstack([x for x in list_qi])
	# np.savetxt("left_hand.txt", left_hand, fmt = "%.2f")
	# np.savetxt("right_hand.txt", right_hand, fmt = "%.2f")
	F_list = np.linalg.lstsq(left_hand, right_hand, rcond=None)[0]
	print(F_list)
	# halt
	A1star = np.zeros(A1.shape)
	for i in range(K):
		A1star += F_list[i] * np.matmul(np.matmul(A1, list_Vnew[i]), list_Vnew[i].T)

	A1star = reconstruct_interpolate(AA1, A1star, A1_MeanMat)

	A1 = A1 + A1_MeanMat

	A1[np.where(AA1.T == 0)] = A1star[np.where(AA1.T == 0)]

	return A1.T


def interpolation_24_v4(AA, AA1, data):
	list_mcol = np.where(~AA1.T.any(axis=0)) 
	list_fcol = np.where(AA1.T.any(axis=0)) 

	length_clip = AA1.shape[1]
	length_sequence = AA.shape[1]	
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[:,0:AA1.shape[1]])

	# ///////////////////////////////////////////////////////
	selected_patch = np.copy(AA[0:length_clip])
	max_diff = 1000000000000
	for i in range(K):
		l = AA1.shape[1]*i+0
		r = AA1.shape[1]*i+AA1.shape[1]
		tmp = np.copy(AA[:,l:r])
		tmp_mse = calculate_mse(tmp, AA1)
		if tmp_mse < max_diff:
			max_diff = tmp_mse
			selected_patch = tmp


	for x in range(10):
		AA = np.hstack((AA, selected_patch))
	B = np.copy(AA)
	B1 = selected_patch
	length_sequence = AA.shape[1]	
	K = length_sequence // length_clip 

	A_MeanVec = np.mean(B, 0)
	A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	A_new = np.copy(B - A_MeanMat)

	A1_MeanVec = np.mean(B1, 0)
	A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	A1_new = np.copy(B1 - A1_MeanMat)
	# np.savetxt("checkA1.txt", A1_MeanMat, fmt = "%.2f")
	# halt

	A1 = np.delete(A1_new.T, list_mcol, 1)
	A = np.copy(A_new.T)
	# axis = 1, get zero row
	
	V = mysvd(np.matmul(A.T, A))
	V = V[:,:AA1.shape[1]]
	list_A = []
	list_A0reduce = []
	list_V0reduce = []

	for i in range(K):
		l = AA1.shape[1]*i+0
		r = AA1.shape[1]*i+AA1.shape[1]
		tmp = np.copy(A[l:r])
		list_A.append(np.copy(tmp))
		tmp_reduce = np.delete(tmp, list_mcol, 1)
		# axis = 0, delete row
		list_A0reduce.append(tmp_reduce)

	V_reduce = np.delete(V, list_mcol, 0)
	V_remain = np.delete(V, list_fcol, 0)
	for i in range(K):
		tmp = mysvd(np.matmul(list_A0reduce[i].T, list_A0reduce[i]))
		list_V0reduce.append(tmp[:,:AA1.shape[1]])

	list_alpha0 = [np.matmul(list_A0reduce[i], list_V0reduce[i]) for i in range(K)]
	list_alpha = [np.matmul(list_A[i], V) for i in range(K)]
	alpha0Talpha0 = np.zeros(np.matmul(list_alpha0[0].T, list_alpha0[0]).shape)
	alpha0T_alpha = np.zeros(np.matmul(list_alpha0[0].T, list_alpha[0]).shape)
	for i in range(K):
		alpha0Talpha0 += np.matmul(list_alpha0[i].T, list_alpha0[i])
		alpha0T_alpha += np.matmul(list_alpha0[i].T, list_alpha[i])

	Fmatrix = np.matmul(np.linalg.inv(alpha0Talpha0), alpha0T_alpha )

	A1reduce = A1
	V1reduce = mysvd(np.matmul(A1reduce.T, A1reduce))[:,:AA1.shape[1]]
	alpha1 = np.matmul(A1reduce, V1reduce)

	A_missing = np.matmul(np.matmul(alpha1, Fmatrix), V_remain.T)
	print("F:")
	check_interpolation(A_missing)
	tmp = np.zeros(AA1.shape)
	tmp[list_fcol] = A1.T
	tmp[list_mcol] = A_missing.T

	tmp = tmp.T + A1_MeanMat.T
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	return result.T


def interpolation_24_v5(AA, AA1, data):
	list_mcol = np.where(~AA1.T.any(axis=0)) 
	list_fcol = np.where(AA1.T.any(axis=0)) 

	length_clip = AA1.shape[1]
	length_sequence = AA.shape[1]	
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[:,0:AA1.shape[1]])

	# ///////////////////////////////////////////////////////
	selected_patch = np.copy(AA[0:length_clip])
	max_diff = 1000000000000
	for i in range(K):
		l = AA1.shape[1]*i+0
		r = AA1.shape[1]*i+AA1.shape[1]
		tmp = np.copy(AA[:,l:r])
		tmp_mse = calculate_mse(tmp, AA1)
		if tmp_mse < max_diff:
			max_diff = tmp_mse
			selected_patch = tmp


	for x in range(10):
		AA = np.hstack((AA, selected_patch))
	B = np.copy(AA)
	B1 = selected_patch
	length_sequence = AA.shape[1]	
	K = length_sequence // length_clip 

	A_MeanVec = np.mean(B, 0)
	A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	A_new = np.copy(B - A_MeanMat)

	A1_MeanVec = np.mean(B1, 0)
	A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	A1_new = np.copy(B1 - A1_MeanMat)

	A1 = np.delete(A1_new.T, list_mcol, 1)
	A = np.copy(A_new.T)
	# axis = 1, get zero row

	V = mysvd(np.matmul(A.T, A))
	V = V[:,:AA1.shape[1]]
	V_reduce = np.delete(V, list_mcol, 0)
	V_remain = np.delete(V, list_fcol, 0)

	A1reduce = A1
	c1 = np.matmul(A1reduce, V_reduce)
	c2 = np.linalg.inv(np.matmul(V_reduce.T, V_reduce))
	A_missing = np.matmul(np.matmul(c1, c2), V_remain.T)
	print("F1:")
	check_interpolation(A_missing)
	tmp = np.zeros(AA1.shape)
	tmp[list_fcol] = A1.T
	tmp[list_mcol] = A_missing.T

	tmp = tmp.T + A1_MeanMat.T
	result = np.copy(AA1.T)
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	return result.T

def interpolation_F(AA, AA1):
	length_clip = AA1.shape[1]
	length_sequence = AA.shape[1]	
	K = length_sequence // length_clip 

	A, A1, A1_MeanMat = deficiency_matrix2(AA, AA1)
	V = mysvd(np.matmul(A.T, A))

	list_A = []
	list_A0 = []
	list_V0 = []
	for i in range(K):
		l = AA1.shape[1]*i+0
		r = AA1.shape[1]*i+AA1.shape[1]
		tmp = np.copy(A[l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1.T == 0)] = 0
		list_A0.append(np.copy(tmp))
		list_V0.append(mysvd(np.matmul(list_A0[-1].T, list_A0[-1])))

	list_alpha0 = [np.matmul(list_A0[i], list_V0[i]) for i in range(K)]
	list_alpha = [np.matmul(list_A[i], V) for i in range(K)]
	alpha0Talpha0 = np.zeros(np.matmul(list_alpha0[0].T, list_alpha0[0]).shape)
	alpha0T_alpha = np.zeros(np.matmul(list_alpha0[0].T, list_alpha[0]).shape)
	for i in range(K):
		alpha0Talpha0 += np.matmul(list_alpha0[i].T, list_alpha0[i])
		alpha0T_alpha += np.matmul(list_alpha0[i].T, list_alpha[i])

	Fmatrix = np.matmul(np.linalg.inv(alpha0Talpha0), alpha0T_alpha )

	V1 = mysvd(np.matmul(A1.T, A1))
	alpha1 = np.matmul(A1, V1)

	A_star = np.matmul(np.matmul(alpha1, Fmatrix), V.T)

	tmp = A_star + A1_MeanMat
	result = np.copy(AA1.T)
	print("F0:")
	check_interpolation(tmp[np.where(AA1.T == 0)])
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	return result.T


def interpolation_24_v6(AA, AA1):
	length_clip = AA1.shape[1]
	length_sequence = AA.shape[1]	
	K = length_sequence // length_clip 
	# B = np.copy(AA)
	# B1 = np.copy(AA[:,0:AA1.shape[1]])

	# # ///////////////////////////////////////////////////////
	# selected_patch = np.copy(AA[0:length_clip])
	# max_diff = 1000000000000
	# for i in range(K):
	# 	l = AA1.shape[1]*i+0
	# 	r = AA1.shape[1]*i+AA1.shape[1]
	# 	tmp = np.copy(AA[:,l:r])
	# 	tmp_mse = calculate_mse(tmp, AA1)
	# 	if tmp_mse < max_diff:
	# 		max_diff = tmp_mse
	# 		selected_patch = tmp


	# for x in range(10):
	# 	AA = np.hstack((AA, selected_patch))
	# B = np.copy(AA)
	# B1 = selected_patch
	# length_sequence = AA.shape[1]	
	# K = length_sequence // length_clip 
	# # ///////////////////////////////////////////////////////


	# A_MeanVec = np.mean(B, 0)
	# A_MeanMat = np.tile(A_MeanVec, (B.shape[0], 1))
	# A_new = np.copy(B - A_MeanMat)

	# A1_MeanVec = np.mean(B1, 0)
	# A1_MeanMat = np.tile(A1_MeanVec, (B1.shape[0], 1))
	# A1_new = np.copy(AA1 - A1_MeanMat)
	# A1_new[np.where(AA1 == 0)] = 0
	# # np.savetxt("checkA1.txt", A1_MeanMat, fmt = "%.2f")
	# # halt
	columnindex = np.where(AA1 == 0)[1]
	columnwithgap = np.unique(columnindex)

	A_MeanVec = np.mean(AA, 0)
	A_MeanMat = np.tile(A_MeanVec, (AA.shape[0], 1))
	A_new = np.copy(AA - A_MeanMat)

	A1_MeanVec = AA1.sum(0) / (AA1 != 0).sum(0)
	A1_MeanMat = np.tile(A1_MeanVec,(AA1.shape[0], 1))
	A1_new = np.copy(AA1 - A1_MeanMat)
	# A1_new[:, columnwithgap] = 0
	A1_new[np.where(AA1 == 0)] = 0
	
	A = np.copy(A_new.T)
	A1 = np.copy(A1_new.T)
	A1_MeanMat = np.copy(A1_MeanMat.T)
	# axis = 1, get zero row

	# count the number of patch in reference
	
	V = mysvd(np.matmul(A.T, A))

	list_A = []
	list_A0 = []
	list_V0 = []
	for i in range(K):
		l = AA1.shape[1]*i+0
		r = AA1.shape[1]*i+AA1.shape[1]
		tmp = np.copy(A[l:r])
		list_A.append(np.copy(tmp))
		tmp[np.where(AA1.T == 0)] = 0
		list_A0.append(np.copy(tmp))
		list_V0.append(mysvd(np.matmul(list_A0[-1].T, list_A0[-1])))

	list_alpha0 = [np.matmul(list_A0[i], list_V0[i]) for i in range(K)]
	list_alpha = [np.matmul(list_A[i], V) for i in range(K)]
	alpha0Talpha0 = np.zeros(np.matmul(list_alpha0[0].T, list_alpha0[0]).shape)
	alpha0T_alpha = np.zeros(np.matmul(list_alpha0[0].T, list_alpha[0]).shape)
	for i in range(K):
		alpha0Talpha0 += np.matmul(list_alpha0[i].T, list_alpha0[i])
		alpha0T_alpha += np.matmul(list_alpha0[i].T, list_alpha[i])

	Fmatrix = np.matmul(np.linalg.inv(alpha0Talpha0), alpha0T_alpha )

	V1 = mysvd(np.matmul(A1.T, A1))
	alpha1 = np.matmul(A1, V1)

	A_star = np.matmul(np.matmul(alpha1, Fmatrix), V.T)

	tmp = A_star + A1_MeanMat
	result = np.copy(AA1.T)
	print("F0:")
	check_interpolation(tmp[np.where(AA1.T == 0)])
	result[np.where(AA1.T == 0)] = tmp[np.where(AA1.T == 0)]
	return result.T


def PCA_PLOS1_F4(AA, AA1):

	combine_matrix = np.vstack((AA, AA1))
	weightScale = 200
	MMweight = 0.02
	[frames, columns] = combine_matrix.shape
	columnindex = np.where(combine_matrix == 0)[1]
	frameindex = np.where(combine_matrix == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	columnwithgap_bypatch = np.copy(columnwithgap - AA.shape[1])
	markerwithgap_bypatch = np.unique(columnwithgap_bypatch // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(combine_matrix, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, combine_matrix.shape[1]//3)
	Data = np.copy(combine_matrix - MeanMat)
	Data[np.where(combine_matrix == 0)] = 0

	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(max(frames-400-len( framewithgap), 0), frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1

	
	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
	
	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
	m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
	m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
	m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
	N_zero = np.multiply(((N_zero-m6) / m5),m33)

	_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
	U_N_nogap = U_N_nogap_VH.T
	print(U_N_nogap.shape)
	_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
	U_N_zero = U_N_zero_VH.T
	print(U_N_zero.shape)
	ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
	U_N_nogap = U_N_nogap[:, :ksmall]
	U_N_zero = U_N_zero[:, :ksmall]
	T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
	reconstructData = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
	
	# reverse normalization
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp[-AA1.shape[0]:,:])
	final_result = np.copy(AA1)
	final_result[np.where(AA1 == 0)] = result[np.where(AA1 == 0)]
	print("checking result PCA long patch:")
	check_interpolation(final_result[np.where(AA1 == 0)])
	return final_result

def PCA_PLOS1_F5(AA, matrix2):

	weightScale = 200
	MMweight = 0.02
	DistalThreshold = 0.5
	MinCumSV = 0.99
	[frames, columns] = AA.shape
	columnindex = np.where(AA == 0)[1]
	frameindex = np.where(AA == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(AA, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
	Data = np.copy(AA - MeanMat)
	Data[np.where(AA == 0)] = 0
	
	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
	
	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
	m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
	m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
	m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
	N_zero = np.multiply(((N_zero-m6) / m5),m33)
	_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
	U_N_nogap = U_N_nogap_VH.T
	_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
	U_N_zero = U_N_zero_VH.T
	ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
	
	# T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
	# T_matrix = np.matmul(np.matmul(np.matmul(U_N_nogap.T, N_nogap.T), N_zero), U_N_zero)
	# Fr = np.matmul(np.matmul(np.matmul(U_N_zero.T, N_zero.T), N_nogap), U_N_nogap)
	# Fl = np.matmul(np.matmul(np.matmul(U_N_zero.T, N_zero.T), N_zero), U_N_zero)
	list_A = []
	list_A0 = []
	list_V = []
	list_V0 = []
	fix_leng = 100
	l = 0
	r = min(fix_leng, N_zero.shape[0])
	K = 0
	ksmall = 0
	while l <= r:
		K += 1
		tmp = np.copy(N_nogap[l:r])
		list_A.append(np.copy(tmp))

		tmp = np.copy(N_zero[l:r])
		list_A0.append(np.copy(tmp))
		
		_, tmp_V0sigma, tmp_V0 = np.linalg.svd(list_A0[-1]/np.sqrt(list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_V0sigma))
		list_V0.append(np.copy(tmp_V0.T))
		_, tmp_Vsigma, tmp_V = np.linalg.svd(list_A[-1]/np.sqrt(list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_Vsigma))
		list_V.append(np.copy(tmp_V.T))
		l += fix_leng
		r += fix_leng
		r = min(r, N_zero.shape[0])

	for i in range(K):
		list_V[i] = list_V[i][:, :ksmall]
		list_V0[i] = list_V0[i][:, :ksmall]

	U_N_nogap = U_N_nogap[:, :ksmall]
	U_N_zero = U_N_zero[:, :ksmall]
	list_alpha0 = [np.matmul(list_A0[i], list_V0[i]) for i in range(K)]
	list_alpha = [np.matmul(list_A[i], list_V[i]) for i in range(K)]
	Fl = np.zeros(np.matmul(list_alpha0[0].T, list_alpha0[0]).shape)
	Fr = np.zeros(np.matmul(list_alpha0[0].T, list_alpha[0]).shape)
	for i in range(K):
		Fl += np.matmul(list_alpha0[i].T, list_alpha0[i])
		Fr += np.matmul(list_alpha0[i].T, list_alpha[i])

	# Fr = np.matmul(np.matmul(np.matmul(list_V0[0].T, list_A0[0].T), list_A[0]), list_V[0])
	# Fl = np.matmul(np.matmul(np.matmul(list_V0[0].T, list_A0[0].T), list_A0[0]), list_V0[0])
	Fl = np.linalg.inv(Fl)
	T_matrix = np.matmul(Fl, Fr)
	# T_matrix = np.matmul(list_V0[0].T , list_V[0])
	reconstructData = np.matmul(np.matmul(np.matmul(M_zero, list_V0[-1]), T_matrix), list_V[-1].T)

	# reverse normalization
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp)

	final_result = np.copy(matrix2)
	final_result[np.where(matrix2 == 0)] = result[np.where(matrix2 == 0)]
	print("T0:")
	check_interpolation(final_result[np.where(matrix2 == 0)])
	return final_result

# implementation regards to Yu 8th
def PCA_PLOS1_F6(AA, matrix2):

	weightScale = 200
	MMweight = 0.02
	DistalThreshold = 0.5
	MinCumSV = 0.99
	[frames, columns] = AA.shape
	columnindex = np.where(AA == 0)[1]
	frameindex = np.where(AA == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(AA, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
	Data = np.copy(AA - MeanMat)
	Data[np.where(AA == 0)] = 0
	
	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
	
	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
	m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
	m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
	m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
	N_zero = np.multiply(((N_zero-m6) / m5),m33)
	_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
	U_N_nogap = U_N_nogap_VH.T
	_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
	U_N_zero = U_N_zero_VH.T
	ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
	
	# T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
	# T_matrix = np.matmul(np.matmul(np.matmul(U_N_nogap.T, N_nogap.T), N_zero), U_N_zero)
	# Fr = np.matmul(np.matmul(np.matmul(U_N_zero.T, N_zero.T), N_nogap), U_N_nogap)
	# Fl = np.matmul(np.matmul(np.matmul(U_N_zero.T, N_zero.T), N_zero), U_N_zero)
	list_A = []
	list_A0 = []
	list_V = []
	list_V0 = []
	list_F = []
	fix_leng = min(100,N_zero.shape[0])
	r = fix_leng
	l = 0
	PQ_size = N_zero.shape[1]
	K = 0
	ksmall = 0
	while l <= r:
		K += 1
		tmp = np.copy(N_nogap[l:r])
		list_A.append(np.copy(tmp))

		tmp = np.copy(N_zero[l:r])
		list_A0.append(np.copy(tmp))
		
		_, tmp_V0sigma, tmp_V0 = np.linalg.svd(list_A0[-1]/np.sqrt(list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_V0sigma))
		list_V0.append(np.copy(tmp_V0.T))
		_, tmp_Vsigma, tmp_V = np.linalg.svd(list_A[-1]/np.sqrt(list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_Vsigma))
		list_V.append(np.copy(tmp_V.T))
		r += fix_leng
		l += fix_leng
		r = min(r, N_zero.shape[0])

	for i in range(K):
		list_V[i] = list_V[i][:, :ksmall]
		list_V0[i] = list_V0[i][:, :ksmall]
		list_F.append(np.matmul(list_V0[i].T, list_V[i]))

	list_qi = []
	list_left = []
	for counter in range(K):
		list_pi = []
		Qi = np.zeros((PQ_size, PQ_size))
		for i in range(K):
			Pi = np.zeros((PQ_size, PQ_size))
			for j in range(K):
				Pi += get_matmulPi8(list_V[counter], list_F[counter].T, list_V0[counter].T, list_A0[j], list_V0[i], list_F[i], list_V[i].T)
			Pi = Pi.reshape(PQ_size * PQ_size, 1)
			list_pi.append(Pi)
			Qi += np.matmul(np.matmul(np.matmul(np.matmul(list_A0[i], list_V0[counter]),list_F[counter]), list_V[counter].T).T, list_A[i])
		Qi = Qi.reshape(PQ_size * PQ_size,1)
		list_qi.append(Qi)
		ls_left = np.hstack([list_pi[i] for i in range(K)])
		list_left.append(ls_left)
	left_hand = np.vstack([x for x in list_left])
	right_hand = np.vstack([x for x in list_qi])

	Fi_list = np.linalg.lstsq(left_hand, right_hand, rcond=None)[0]
	print(Fi_list)
	# for x in range(K):
	# 	Fi_list[x] = (1.0 - Fi_list[x])/(K-1)
	# print(Fi_list)
	sum_error = 0
	for i in range(K):
		tmp_result = np.zeros(list_A0[i].shape)
		for j in range(K):
			tmp_result += Fi_list[j] * np.matmul(np.matmul(np.matmul(list_A0[i], list_V0[j]),list_F[j]), list_V[j].T)
		sum_error += Fi_list[i] * euclid_dist(list_A[i], tmp_result)

	print("checking result")
	print(sum_error)
	print("ending")
	reconstructData = np.copy(M_zero)
	tmp_result = np.zeros(M_zero[-fix_leng:].shape)
	for i in range(K):
		tmp_result += Fi_list[i] * np.matmul(np.matmul(np.matmul(M_zero[-fix_leng:], list_V0[i]),list_F[i]), list_V[i].T)

	reconstructData[-fix_leng:] = tmp_result
	# reverse normalization
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp)

	final_result = np.copy(matrix2)
	final_result[np.where(matrix2 == 0)] = result[np.where(matrix2 == 0)]
	print("T0:")
	check_interpolation(final_result[np.where(matrix2 == 0)])
	return final_result

def PCA_PLOS1_F7(AA, matrix2):

	weightScale = 200
	MMweight = 0.02
	DistalThreshold = 0.5
	MinCumSV = 0.99
	[frames, columns] = AA.shape
	columnindex = np.where(AA == 0)[1]
	frameindex = np.where(AA == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	Data_without_gap = np.delete(AA, columnwithgap, 1)
	mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
	columnWithoutGap = Data_without_gap.shape[1]

	x_index = [x for x in range(0, columnWithoutGap, 3)]
	mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

	y_index = [x for x in range(1, columnWithoutGap, 3)]
	mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

	z_index = [x for x in range(2, columnWithoutGap, 3)]
	mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

	joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
	MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
	Data = np.copy(AA - MeanMat)
	Data[np.where(AA == 0)] = 0
	
	# calculate weight vector 
	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] ,get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						tmp = 0
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith
	weight_vector = np.min(weight_vector, 0)
	weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
	weight_vector[markerwithgap] = MMweight
	M_zero = np.copy(Data)
	# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
	N_nogap = np.delete(Data, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0
	
	mean_N_nogap = np.mean(N_nogap, 0)
	mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

	mean_N_zero = np.mean(N_zero, 0)
	mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
	stdev_N_no_gaps = np.std(N_nogap, 0)
	stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


	m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
	m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
	
	column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
	column_weight = column_weight.reshape((1, column_weight.shape[0]))
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
	m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
	m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
	m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

	M_zero = np.multiply(((M_zero-m1) / m2),m3)
	N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
	N_zero = np.multiply(((N_zero-m6) / m5),m33)
	_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
	U_N_nogap = U_N_nogap_VH.T
	_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
	U_N_zero = U_N_zero_VH.T
	ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
	
	# T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
	# T_matrix = np.matmul(np.matmul(np.matmul(U_N_nogap.T, N_nogap.T), N_zero), U_N_zero)
	# Fr = np.matmul(np.matmul(np.matmul(U_N_zero.T, N_zero.T), N_nogap), U_N_nogap)
	# Fl = np.matmul(np.matmul(np.matmul(U_N_zero.T, N_zero.T), N_zero), U_N_zero)
	list_A = []
	list_A0 = []
	list_V = []
	list_V0 = []
	list_F = []
	fix_leng = min(100,N_zero.shape[0])
	r = fix_leng
	l = 0
	PQ_size = N_zero.shape[1]
	K = 0
	ksmall = 0
	while l <= r:
		K += 1
		tmp = np.copy(N_nogap[l:r])
		list_A.append(np.copy(tmp))

		tmp = np.copy(N_zero[l:r])
		list_A0.append(np.copy(tmp))
		
		_, tmp_V0sigma, tmp_V0 = np.linalg.svd(list_A0[-1]/np.sqrt(list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_V0sigma))
		list_V0.append(np.copy(tmp_V0.T))
		_, tmp_Vsigma, tmp_V = np.linalg.svd(list_A[-1]/np.sqrt(list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_Vsigma))
		list_V.append(np.copy(tmp_V.T))
		r += fix_leng
		l += fix_leng
		r = min(r, N_zero.shape[0])

	for i in range(K):
		list_V[i] = list_V[i][:, :ksmall]
		list_V0[i] = list_V0[i][:, :ksmall]
		list_F.append(np.matmul(list_V0[i].T, list_V[i]))
	current_weight = [1.0/K]*K
	max_looper = 50
	down_rate = 0.25
	pre_err = 1000
	best_err = 1000
	best_weight = [1.0/K]*K
	point_list = [0] * K
	diff_lim = 1.0 / (K*K)
	print(diff_lim)
	for loop in range(max_looper):
		print("looper: ", loop)
		list_qi = []
		list_left = []
		for counter in range(K):
			list_pi = []
			Qi = np.zeros((PQ_size, PQ_size))
			for i in range(K):
				Pi = np.zeros((PQ_size, PQ_size))
				for j in range(K):
					Pi += current_weight[j] * get_matmulPi8(list_V[counter], list_F[counter].T, list_V0[counter].T, list_A0[j], list_V0[i], list_F[i], list_V[i].T)
				Pi = Pi.reshape(PQ_size * PQ_size, 1)
				list_pi.append(Pi)
				Qi += current_weight[i] * np.matmul(np.matmul(np.matmul(np.matmul(list_A0[i], list_V0[counter]),list_F[counter]), list_V[counter].T).T, list_A[i])
			Qi = Qi.reshape(PQ_size * PQ_size,1)
			list_qi.append(Qi)
			ls_left = np.hstack([list_pi[i] for i in range(K)])
			list_left.append(ls_left)
		left_hand = np.vstack([x for x in list_left])
		right_hand = np.vstack([x for x in list_qi])

		Fi_list = np.linalg.lstsq(left_hand, right_hand, rcond=None)[0]
		for x in range(K):
			point_list[x] = float(Fi_list[x])
		print(Fi_list)
		for x in range(K):
			Fi_list[x] = (1.0 - Fi_list[x])/(K-1)
		print("impact on weight")
		print(Fi_list)
		# print(point_list)
		print("end")
		max_sample = get_max_array(Fi_list, current_weight)
		min_sample = get_min_array(Fi_list, current_weight)
		print(max_sample, " ", min_sample)
		print(diff_lim, " ", abs(point_list[max_sample] - point_list[min_sample]), " ", point_list[max_sample], " ", point_list[min_sample])
		if max_sample == min_sample or abs(point_list[max_sample] - point_list[min_sample]) <= diff_lim:
			print("stop by reach limit")
			break
		current_weight[max_sample] += current_weight[min_sample]* down_rate
		current_weight[min_sample] -= current_weight[min_sample]* down_rate
		if current_weight[min_sample] <= 0.0099:
			current_weight[max_sample] += current_weight[min_sample]
			current_weight[min_sample] = 0

		print(current_weight)
		sum_error = 0
		for i in range(K):
			tmp_result = np.zeros(list_A0[i].shape)
			for j in range(K):
				tmp_result += current_weight[j] * np.matmul(np.matmul(np.matmul(list_A0[i], list_V0[j]),list_F[j]), list_V[j].T)
			sum_error += current_weight[i] * euclid_dist(list_A[i], tmp_result)

		print("checking result")
		print(sum_error)
		print(best_err)
		print("ending")
		if best_err > sum_error:
			best_err = sum_error
			for x in range(K):
				best_weight[x] = current_weight[x]
		print(best_weight)
		if pre_err < sum_error - 0.5: 
			print("break by over regulated error")
			break
		pre_err = sum_error
	print("final weight")
	print(best_weight)
	reconstructData  = np.zeros(M_zero.shape)
	for i in range(K):
		reconstructData += best_weight[i] * np.matmul(np.matmul(np.matmul(M_zero, list_V0[i]),list_F[i]), list_V[i].T)

	# reverse normalization
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
	m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
	reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
	tmp = reconstructData + MeanMat
	result = np.copy(tmp)

	final_result = np.copy(matrix2)
	final_result[np.where(matrix2 == 0)] = result[np.where(matrix2 == 0)]
	print("T0:")
	check_interpolation(final_result[np.where(matrix2 == 0)])
	return final_result


def PCA_PLOS1_no_normalize(AA, AA1):
	combine_matrix = np.vstack((AA, AA1))
	[frames, columns] = combine_matrix.shape
	columnindex = np.where(combine_matrix == 0)[1]
	frameindex = np.where(combine_matrix == 0)[0]
	columnwithgap = np.unique(columnindex)
	markerwithgap = np.unique(columnwithgap // 3)
	framewithgap = np.unique(frameindex)
	M_zero = np.copy(combine_matrix)

	N_nogap = np.delete(combine_matrix, framewithgap, 0)
	N_zero = np.copy(N_nogap)
	N_zero[:,columnwithgap] = 0

	_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
	U_N_nogap = U_N_nogap_VH.T
	print(U_N_nogap.shape)
	_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
	U_N_zero = U_N_zero_VH.T
	print(U_N_zero.shape)
	ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
	print("///////////////////////////////: ", ksmall)
	U_N_nogap = U_N_nogap[:, :ksmall]
	U_N_zero = U_N_zero[:, :ksmall]
	T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
	reconstructData = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
	
	# reverse normalization
	tmp = np.copy(reconstructData)
	result = np.copy(tmp[-AA1.shape[0]:,:])
	final_result = np.copy(AA1)
	final_result[np.where(AA1 == 0)] = result[np.where(AA1 == 0)]
	print("checking result PCA long patch:")
	check_interpolation(final_result[np.where(AA1 == 0)])
	return final_result

def get_max_array(arrayA, weight):
	result = 0
	for x in range(len(arrayA)):
		if arrayA[result] < arrayA[x] : result = x
	return result

def get_min_array(arrayA, weight):

	result = 0 
	value = 0
	for x in range(len(arrayA)):
		if arrayA[x] > value: 
			result = x
			value = arrayA[x]

	for x in range(len(arrayA)):
		if arrayA[result] > arrayA[x] and arrayA[x] > 0.001 and weight[x] > 0.01 : result = x
	return result


def interpolation(A1, IUT, TTU1TA1R, VTI, A1V1FR, A1_MeanMat):
	A_new = np.copy(A1)
	A = np.concatenate((IUT, VTI), axis=0)
	B = np.concatenate((TTU1TA1R, A1V1FR), axis=0)
	X = np.linalg.lstsq(A, B, rcond=None)
	XX =  X[0].reshape(A1.shape[0],A1.shape[1])
	Astar = reconstruct_interpolate(A1.T, X[0].reshape(A1.shape[0],A1.shape[1]), A1_MeanMat.T)
	A_new[np.where(A1 == 0)] = Astar[np.where(A1 == 0)]
	return A_new


def check_interpolation(matrix):
	if np.sum(np.abs(matrix)) <= 0.000001:
		print("can't interpolated")
		return
	print("interpolated successfully")
	return

def euclid_dist(X, Y):
	XX = np.asarray(X)
	YY = np.asarray(Y)
	return np.sqrt(np.sum(np.square(XX - YY)))


def calculate_mse(X, Y):
	mse = (np.square(X - Y)).mean()
	mse = (np.sqrt(mse))
	return mse

def calculate_mse_patch(X, Y):
	XX = np.copy(X[np.where(Y != 0)])
	YY = np.copy(Y[np.where(Y != 0)])
	mse = (np.square(XX - YY)).mean()
	mse = (np.sqrt(mse))
	return mse

def calculate_mse_matrix(X):
	mse = (np.square(X))
	sum_distance = 0
	for x in range(len(X)//3):
		sum_distance += np.sqrt(mse[x*3] + mse[x*3+1] + mse[x*3+2])
	error =  sum_distance / (len(X)/3)
	return error


def calculate_mse_matrix_Yu(X, divide = 1):
	mse = np.sqrt(np.sum(np.square(X)))
	mse = mse /(len(X))
	return mse


def calculate_mae_matrix(X):
	error_sum = np.sum(np.abs(X))
	mse = np.sum(np.square(X))
	print("debug")
	print("mse: ",mse)
	print("mae: ",error_sum)
	print("end")
	return error_sum / len(X)

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


def get_random_joint3D(A, length, num_joint_missing, frames):
	number_frame_missing = 15
	AA = np.copy(A)
	# l = [x for x in range(length)]
	# missing_frame_arr = random.sample(l, number_frame_missing)
	missing_frame_arr = frames
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
		for x in range(0, length-0):
			AA[x, index*2] = 0
			AA[x, index*2+1] = 0
	return AA


def get_remove_row3D(A, length, num_row_missing):
	AA = np.copy(A)
	arr = random.sample(arg.missing_row_arr, num_row_missing)
	for index in arr:
		for x in range(0, length-0):
			AA[x, index*3] = 0
			AA[x, index*3+1] = 0
			AA[x, index*3+2] = 0
	return AA

def get_joint_over_Aframe(A, joint_quantity, frame_index):
	AA = np.copy(A)
	for x in range(joint_quantity):
		AA[frame_index, x*2] = 0
		AA[frame_index, x*2+1] = 0
	return AA


def get_joint_over_Acolumn(A, frame_quantity, joint_index):
	AA = np.copy(A)
	for x in range(frame_quantity):
		AA[x, joint_index*2] = 0
		AA[x, joint_index*2+1] = 0
	return AA


def setting_rank(eigen_vector):
	minCumSV = 0.99
	sum_list = np.sum(eigen_vector)
	current_sum = 0
	for x in range(len(eigen_vector)):
		current_sum += eigen_vector[x]
		if current_sum > minCumSV * sum_list:
			return x+1
	return len(eigen_vector)