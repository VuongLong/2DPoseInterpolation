import numpy as np

def diag_matrix(value, k):
	value_array = [value]*k
	return np.diag(value_array)

def matmul_list(matrix_list):
	number_matrix = len(matrix_list)
	result = np.copy(matrix_list[0])
	for i in range(1, number_matrix):
		result = np.matmul(result, matrix_list[i])
	return result

def summatrix_list(matrix_list):
	number_matrix = len(matrix_list)
	result = np.copy(matrix_list[0])
	for i in range(1, number_matrix):
		result = result + matrix_list[i]
	return result


def MSE(A, B, missing_map):
	return np.sum(np.abs(A - B) * (1-missing_map)) / np.sum(missing_map)

def ARE(predict, original):
	return np.mean(np.abs((predict - original) / original))

def mysvd(dataMat):
	U, Sigma, VT = np.linalg.svd(dataMat)
	return U

def remove_joint(data):
	list_del = []
	list_del_joint = [5, 9, 14, 18]

	for x in list_del_joint:
		list_del.append(x*3)
		list_del.append(x*3+1)
		list_del.append(x*3+2)
	data = np.delete(data, list_del, 1)
	#print('removed joints data', data.shape)
	return data 

def euclid_dist(X, Y):
	XX = np.asarray(X)
	YY = np.asarray(Y)
	return np.sqrt(np.sum(np.square(XX - YY)))

def get_zero(matrix):
	counter = 0
	for x in matrix:
		if x > 0.01: counter += 1
	return counter

def get_point(Data, frame, joint):
	point = [ Data[frame, joint*3] , Data[frame, joint*3+1] , Data[frame, joint*3+2 ]]
	return point


def read_tracking_data3D(data_dir, patch):
	print("reading source: ", data_dir, " patch: ", patch)

	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line.split(',')
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	#print('original data', Tracking3D.shape)

	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D[patch[0]: patch[1]]
	#print('patch data', Tracking3D.shape)

	Tracking3D = remove_joint(Tracking3D)
	restore = np.copy(Tracking3D)
	return Tracking3D, restore

def read_tracking_data3D_without_RJ(data_dir, patch):
	#print("reading source: ", data_dir, " patch: ", patch)

	Tracking3D = []
	f=open(data_dir, 'r')
	for line in f:
		elements = line.split(' ')
		Tracking3D.append(list(map(float, elements)))
	f.close()

	Tracking3D = np.array(Tracking3D) # list can not read by index while arr can be
	Tracking3D = np.squeeze(Tracking3D)
	#print('original data', Tracking3D.shape)

	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D[patch[0]: patch[1]]
	#print('patch data', Tracking3D.shape)
	
	restore = np.copy(Tracking3D)
	return Tracking3D, restore

def setting_rank(eigen_vector):
	minCumSV = 0.99
	current_sum = 0
	sum_list = np.sum(eigen_vector)
	for x in range(len(eigen_vector)):
		current_sum += eigen_vector[x]
		if current_sum > minCumSV * sum_list:
			return x+1
	return len(eigen_vector)


def get_homogeneous_solve(A, eps=1e-15):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T

def solution(U):
    # find the eigenvalues and eigenvector of U(transpose).U
    e_vals, e_vecs = np.linalg.eig(np.dot(U.T, U))  
    print(e_vals)
    # extract the eigenvector (column) associated with the minimum eigenvalue
    return e_vecs[:, np.argmin(e_vals)] 


def compute_norm(combine_matrix, downsample = False, gap_strategies = False):
	AA = np.copy(combine_matrix)
	weightScale = 200
	MMweight = 0.02
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
	frame_range_start = 0
	if downsample:
		frame_range_start = max(frames-400-len( framewithgap), 0)

	weight_matrix = np.zeros((frames, columns//3))
	weight_matrix_coe = np.zeros((frames, columns//3))
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frame_range_start, frames):
			valid = True
			if euclid_dist([0, 0, 0] , get_point(Data, i, markerwithgap[x])) == 0 :
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
	if gap_strategies == False:
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
	
	m7 = np.ones((Data.shape[0],1))*mean_N_nogap
	
	return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat]

def compute_weight_vect_norm(markerwithgap, Data):
	weightScale = 200
	MMweight = 0.02
	AA = np.copy(Data)
	[frames, columns] = AA.shape
	marker = markerwithgap[0]
	frameindex = np.where(AA[:, marker*3+1] == 0)[0]
	framewithgap = np.unique(frameindex)

	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] , get_point(Data, i, markerwithgap[x])) == 0 :
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)
						if euclid_dist(point2, [0, 0, 0]) != 0:
							weight_matrix[i][j] = euclid_dist(point2, point1)
							weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith

	weight_vector = np.min(weight_vector, 0)
	return weight_vector

def compute_weight_vect_norm_v2(markerwithgap, Data):
	weightScale = 200
	MMweight = 0.02
	AA = np.copy(Data)
	[frames, columns] = AA.shape
	marker = markerwithgap[0]
	frameindex = np.where(AA[:, marker*3+1] == 0)[0]
	framewithgap = np.unique(frameindex)
	weight_vector = np.zeros((len(markerwithgap), columns//3))
	for x in range(len(markerwithgap)):
		weight_matrix = np.zeros((frames, columns//3))
		weight_matrix_coe = np.zeros((frames, columns//3))
		for i in range(frames):
			valid = True
			if euclid_dist([0, 0, 0] , get_point(Data, i, markerwithgap[x])) == 0:
				valid = False
			if valid:
				for j in range(columns//3):
					if j != markerwithgap[x]:
						point1 = get_point(Data, i, markerwithgap[x])
						point2 = get_point(Data, i, j)						
						weight_matrix[i][j] = euclid_dist(point2, point1)
						weight_matrix_coe[i][j] = 1

		sum_matrix = np.sum(weight_matrix, 0)
		sum_matrix_coe = np.sum(weight_matrix_coe, 0)
		weight_vector_ith = sum_matrix / sum_matrix_coe
		weight_vector_ith[markerwithgap[x]] = 0
		weight_vector[x] = weight_vector_ith

	weight_vector = np.min(weight_vector, 0)
	return weight_vector

def check_vector_overlapping(vector_check):
	if len(np.where(vector_check == 0)[0]) > 0:
		return True
	return False

def get_list_frameidx_patch(K_patch, fix_leng, frame_start):
	list_frameidx_patch = []
	for patch_number in range(K_patch):
		patch_start_frame = patch_number * fix_leng + 0
		patch_end_frame = patch_start_frame + fix_leng
		list_frameidx_patch.append([patch_start_frame, patch_end_frame])
	return list_frameidx_patch
