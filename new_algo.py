from algorithm import *

def adaboost(AA, matrix2):
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

	# start adaboost
	Theshold = 0.5
	iteration_limit = 100
	ditribution = [1.0/K]
	error = 0
	list_function = []
	# completing
	for iteration in range(iteration_limit):
		model = regression_model()
		error_sample = []
		for sample in range(K):
			sample_predict = model.interpolation()
			error_sample.append(sample_predict, original_sample)
		
	# completing
	# end 
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
	return final_resultreturn result