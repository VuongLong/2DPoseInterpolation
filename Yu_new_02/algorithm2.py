import numpy as np
import random
from Yu_new_02.preprocess import normalize
from Yu_new_02.utils import *

class Interpolation16th_F():
	def __init__(self, reference_matrix, missing_matrix):
		self.A1 = np.copy(missing_matrix)
		self.AN_F = np.copy(reference_matrix)
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.fix_leng = missing_matrix.shape[0]
		self.normed_matries, self.reconstruct_matries = self.normalization()
		self.K = 0
		self.list_A = []
		self.list_A0 = []
		self.list_V = []
		self.list_V0 = []
		self.list_F = []	
		self.list_alpha = []
		self.compute_svd()
		self.inner_compute_alpha()

	def normalization(self):
		AA = np.copy(self.combine_matrix)
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
		
		# ///////////////////////////////////////////////
		# this peice of code to return result of original PCA method, return result into self.debug
		# this is for comparision with PCA original
		# ///////////////////////////////////////////////


		# _, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
		# U_N_nogap = U_N_nogap_VH.T
		# _, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
		# U_N_zero = U_N_zero_VH.T
		# ksmall = max(get_zero(Sigma_zero), get_zero(Sigma_nogap))
		# U_N_nogap = U_N_nogap[:, :ksmall]
		# U_N_zero = U_N_zero[:, :ksmall]
		# T_matrix = np.matmul(U_N_nogap.T , U_N_zero)
		# reconstructData = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
		
		# # reverse normalization
		# m7 = np.ones((Data.shape[0],1))*mean_N_nogap
		# m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
		# m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		# reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)
		# tmp = reconstructData + MeanMat
		# result = np.copy(tmp)

		# final_result = np.copy(self.combine_matrix)
		# final_result[np.where(self.combine_matrix == 0)] = result[np.where(self.combine_matrix == 0)]
		# self.debug = np.copy(final_result[-self.fix_leng:])
		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat]

	def compute_svd(self):
		M_zero = self.normed_matries[0]
		N_nogap = self.normed_matries[1]
		N_zero = self.normed_matries[2]
		# r = N_zero.shape[0]
		# l = r - self.fix_leng
		r = self.fix_leng
		l = 0
		add_small_patch = False
		ksmall = 0
		while l <= r:
			if r - l < 15: 
				break
			if r - l < self.fix_leng:
				add_small_patch = True
			self.K += 1
			tmp = np.copy(N_nogap[l:r])
			self.list_A.append(np.copy(tmp))

			tmp = np.copy(N_zero[l:r])
			self.list_A0.append(np.copy(tmp))
			
			_, tmp_V0sigma, tmp_V0 = np.linalg.svd(self.list_A0[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_V0sigma))
			self.list_V0.append(np.copy(tmp_V0.T))
			_, tmp_Vsigma, tmp_V = np.linalg.svd(self.list_A[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
			ksmall = max(ksmall, get_zero(tmp_Vsigma))
			self.list_V.append(np.copy(tmp_V.T))
			r += self.fix_leng
			r = min(r, N_zero.shape[0])
			l += self.fix_leng


		# ///////////////////////////////////////////////
		# this peice of code to add combine matrix to list
		# /////////////////////begin//////////////////////////
		self.list_A.append(np.copy(N_nogap))
		self.list_A0.append(np.copy(N_zero))
		_, tmp_V0sigma, tmp_V0 = np.linalg.svd(self.list_A0[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_V0sigma))
		self.list_V0.append(np.copy(tmp_V0.T))
		_, tmp_Vsigma, tmp_V = np.linalg.svd(self.list_A[-1]/np.sqrt(self.list_A0[-1].shape[0]-1), full_matrices = False)
		ksmall = max(ksmall, get_zero(tmp_Vsigma))
		self.list_V.append(np.copy(tmp_V.T))
		self.K += 1
		Vmatrix_size = self.list_V[-2].shape[1]
		ksmall = min(ksmall, Vmatrix_size)

		print("K info: ", self.K)
		# /////////////////////end////////////////////////// 
		for i in range(self.K):
			self.list_V[i] = self.list_V[i][:, :ksmall]
			self.list_V0[i] = self.list_V0[i][:, :ksmall]
			self.list_F.append(np.matmul(self.list_V0[i].T, self.list_V[i]))

		# self.weight_sample = [1.0/self.K]*self.K
		if add_small_patch:
			self.weight_sample = [0.2/self.K]*self.K
			self.weight_sample[-1] += 0.4
			self.weight_sample[-2] += 0.4
		else:
			self.weight_sample = [0.6/self.K]*self.K
			self.weight_sample[-1] += 0.4

	def inner_compute_alpha(self):
		# build list_alpha
		list_Q = []
		for iloop in range(self.K):
			for kloop in range(self.K):
				qi = matmul_list([self.list_V[kloop], self.list_F[kloop].T, self.list_V0[iloop].T, self.list_A0[iloop].T, 
						diag_matrix(self.weight_sample[iloop], self.list_A[iloop].shape[0]), self.list_A[iloop]])
				list_Q.append(qi)
		tmp_matrix = np.zeros(list_Q[-1].shape)
		for x in list_Q:
			tmp_matrix += x
		tmp_matrix = tmp_matrix.reshape(tmp_matrix.shape[0] * tmp_matrix.shape[1], 1)
		right_hand = np.copy(tmp_matrix)

		list_P = []
		for jloop in range(self.K):
			list_Pijk = []
			for iloop in range(self.K):
				for kloop in range(self.K):
					Pijk = matmul_list([self.list_V[kloop], self.list_F[kloop].T, self.list_V0[iloop].T, self.list_A0[iloop].T, 
							diag_matrix(self.weight_sample[iloop], self.list_A0[iloop].shape[0]), self.list_A0[iloop], 
							self.list_V0[iloop], self.list_F[jloop], self.list_V[jloop].T ])
					list_Pijk.append(Pijk)
			tmp_matrix = np.zeros(list_Pijk[-1].shape)
			for x in list_Pijk:
				tmp_matrix += x
			shape_x, shape_y = tmp_matrix.shape[0], tmp_matrix.shape[1]
			tmp_matrix = tmp_matrix.reshape(shape_x * shape_y, 1)
			list_P.append(np.copy(tmp_matrix))

		left_hand = np.hstack([ x for x in list_P])
		tmp_alpha = np.linalg.lstsq(np.matmul(left_hand.T, left_hand), np.matmul(left_hand.T, right_hand), rcond = None)[0]
		sum_alpha = np.sum(tmp_alpha)
		# self.list_alpha = np.copy(tmp_alpha/sum_alpha)
		self.list_alpha = np.copy(tmp_alpha)
		# 
		# debug alpha
		# 
		# for x in range(len(self.list_alpha)):
		# 	self.list_alpha[x] = 0
		# self.list_alpha[-2] = 0.5
		# self.list_alpha[-1] = 0.5
		return 0

	def interpolate_missing(self):
		M_zero = self.normed_matries[0]
		reconstructData = np.copy(M_zero)
		tmp_result = np.zeros(self.A1.shape)
		for i in range(self.K):
			tmp_result += diag_matrix(self.list_alpha[i], self.list_F[i].shape[0]) * np.matmul(np.matmul(np.matmul(M_zero[-self.fix_leng:, :], self.list_V0[i]), 
											self.list_F[i]), self.list_V[i].T)
		reconstructData[-self.fix_leng:] = tmp_result
		m8 = np.ones((reconstructData.shape[0],1))*self.reconstruct_matries[1]
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), self.reconstruct_matries[2])
		reconstructData = self.reconstruct_matries[0] + (np.multiply(reconstructData, m8) / m3)
		tmp = reconstructData + self.reconstruct_matries[3]
		result = np.copy(tmp)

		final_result = np.copy(self.combine_matrix)
		final_result[np.where(self.combine_matrix == 0)] = result[np.where(self.combine_matrix == 0)]
		return final_result[-self.A1.shape[0]:,:]

	def interpolate_sample(self):
		list_error = []
		for sample_idx in range(self.K):
			current_missing_sample = self.list_A0[sample_idx]
			current_original_sample = self.list_A[sample_idx]
			tmp_result = np.zeros(current_missing_sample.shape)
			for i in range(self.K):
				tmp_result += diag_matrix(self.list_alpha[i], self.list_F[i].shape[0]) * np.matmul(np.matmul(np.matmul(current_missing_sample, 
												self.list_V0[i]), self.list_F[i]), self.list_V[i].T)
			result_sample = np.copy(current_missing_sample)
			result_sample[np.where(current_missing_sample == 0)] = tmp_result[np.where(current_missing_sample == 0)]
			list_error.append(ARE(result_sample, current_original_sample))
		return list_error


	def get_weight(self):
		return self.weight_sample

	def set_weight(self, weight):
		# weight must be a new object
		for x in range(self.K):
			self.weight_sample[x] = weight[x]

	def get_number_sample(self):
		return self.K

	def get_alpha(self):
		return self.list_alpha


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

class Interpolation_T():
	def __init__(self, reference_matrix, missing_matrix, norm):
		self.A1 = np.copy(missing_matrix)
		self.AN = np.copy(reference_matrix)
		self.K = int(reference_matrix.shape[0] / missing_matrix.shape[0])
		# self.fix_leng = 100
		self.fix_leng = missing_matrix.shape[0]
		self.AN0 = self.create_AN0()
		self.compute_svd()
		self.T_matrix = self.T
		self.result_nonorm = self.interpolate_missing()
		if norm:
			self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
			self.normed_matries, self.reconstruct_matries = self.normalization()
			self.A1 = np.copy(self.normed_matries[0])
			self.AN = np.copy(self.normed_matries[1])
			self.AN0 = np.copy(self.normed_matries[2])
			self.compute_svd()
			self.result_norm = self.interpolate_missing()
			self.de_normalization()


	def create_AN0(self):
		p_AN = self.AN
		list_A0 = []
		for patch_number in range(self.K):
			l = patch_number * self.fix_leng
			r = (patch_number+1) * self.fix_leng
			tmp = np.copy(p_AN[l:r])
			tmp[np.where(self.A1 == 0)] = 0
			list_A0.append(np.copy(tmp))
		p_AN0 = np.vstack(list_A0)
		return p_AN0

	def normalization(self):
		normed_matries, reconstruct_matries = compute_norm(self.combine_matrix)
		return normed_matries, reconstruct_matries


	def de_normalization(self):
		M_zero = self.normed_matries[0]
		reconstructData = np.copy(self.result_norm)
		m8 = np.ones((reconstructData.shape[0],1))*self.reconstruct_matries[1]
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), self.reconstruct_matries[2])
		reconstructData = self.reconstruct_matries[0] + (np.multiply(reconstructData, m8) / m3)
		tmp = reconstructData + self.reconstruct_matries[3]
		result = np.copy(tmp)

		final_result = np.copy(self.combine_matrix)
		final_result[np.where(self.combine_matrix == 0)] = result[np.where(self.combine_matrix == 0)]
		self.result_norm = np.copy(final_result[-self.fix_leng:])
		return self.result_norm


	def compute_svd(self):
		p_AN = self.AN
		p_AN0 = self.AN0
		list_A0 = []
		list_A = []

		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A.append(np.copy(p_AN[l:r]))
			list_A0.append(np.copy(p_AN0[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)

		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		list_matrix = []
		for patch_number in range(self.K):
			tmp = matmul_list([self.UN0.T, list_A0[patch_number].T, list_A0[patch_number], self.UN0])
			list_matrix.append(tmp)
		tmp_sum = summatrix_list(list_matrix)
		_, dtmp, _ = np.linalg.svd(tmp_sum)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma),setting_rank(dtmp))
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]

		list_left = []
		list_right = []
		for patch_number in range(self.K):
			tmpL = matmul_list([self.UN0.T, list_A0[patch_number].T, list_A[patch_number], self.UN])
			list_left.append(tmpL)
			tmpR = matmul_list([self.UN0.T, list_A0[patch_number].T, list_A0[patch_number], self.UN0])
			list_right.append(tmpR)

		left_form = summatrix_list(list_left)
		right_form = summatrix_list(list_right)
		self.T = np.matmul(np.linalg.inv(left_form), right_form)
		# self.T = np.matmul(self.UN0.T, self.UN)
		return self.T


	def interpolate_missing(self):
		# self.normalization()

		result = matmul_list( [self.A1, self.UN0, self.T, self.UN.T] )
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]

		# self.de_normalization()
		return result


class interpolation_weighted_T():
	def __init__(self, reference_matrix, missing_matrix, norm):
		self.A1 = np.copy(missing_matrix)
		self.AN = np.copy(reference_matrix)
		self.K = int(self.AN.shape[0] / missing_matrix.shape[0])
		# self.fix_leng = 100
		self.fix_leng = missing_matrix.shape[0]
		self.AN0 = self.create_AN0()
		self.compute_svd()
		self.result_nonorm = self.interpolate_missing()
		if norm:
			self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
			self.normed_matries, self.reconstruct_matries = self.normalization()
			self.A1 = np.copy(self.normed_matries[0])
			self.AN = np.copy(self.normed_matries[1])
			self.AN0 = np.copy(self.normed_matries[2])
			self.K = int(self.AN.shape[0] / self.fix_leng)
			self.compute_svd()
			self.result_norm = self.interpolate_missing()
			self.de_normalization()

	def create_AN0(self):
		p_AN = self.AN
		list_A0 = []
		for patch_number in range(self.K):
			l = patch_number * self.fix_leng
			r = (patch_number+1) * self.fix_leng
			tmp = np.copy(p_AN[l:r])
			tmp[np.where(self.A1 == 0)] = 0
			list_A0.append(np.copy(tmp))
		p_AN0 = np.vstack(list_A0)
		return p_AN0

	def normalization(self):
		normed_matries, reconstruct_matries = compute_norm(self.combine_matrix)
		return normed_matries, reconstruct_matries


	def de_normalization(self):
		M_zero = self.normed_matries[0]
		reconstructData = np.copy(self.result_norm)
		m8 = np.ones((reconstructData.shape[0],1))*self.reconstruct_matries[1]
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), self.reconstruct_matries[2])
		reconstructData = self.reconstruct_matries[0] + (np.multiply(reconstructData, m8) / m3)
		tmp = reconstructData + self.reconstruct_matries[3]
		result = np.copy(tmp)

		final_result = np.copy(self.combine_matrix)
		final_result[np.where(self.combine_matrix == 0)] = result[np.where(self.combine_matrix == 0)]
		self.result_norm = np.copy(final_result[-self.fix_leng:])
		return self.result_norm



	def compute_svd(self):
		p_AN = self.AN
		p_AN0 = self.AN0
		list_A0 = []
		list_A = []

		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A.append(np.copy(p_AN[l:r]))
			list_A0.append(np.copy(p_AN0[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)

		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma))
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		self.list_Ti = []

		for patch_number in range(self.K):
			AiUN = np.matmul(list_A[patch_number], self.UN)
			Ai0UN0 = np.matmul(list_A0[patch_number], self.UN0)
			
			# X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
			# self.list_Ti.append(np.copy(X[0]))

			self.list_Ti.append(np.matmul(self.UN.T, self.UN0))
			
			# UN0T_Ai0T = np.matmul(self.UN0.T, list_A0[patch_number].T)
			# left_form = np.matmul(UN0T_Ai0T, AiUN)
			# right_form = np.matmul(UN0T_Ai0T, Ai0UN0)
			# tmpT = np.matmul(np.linalg.inv(left_form), right_form)
			# self.list_Ti.append(tmpT)
		
		# compute weight
	
		list_left_matrix = []
		for patch_number in range(self.K):
			current_patch = np.matmul(list_A[patch_number], self.UN) - matmul_list(
				[list_A0[patch_number], self.UN0, self.list_Ti[patch_number]])
			for column in range(ksmall):
				for clm in range(ksmall):
					tmp = np.multiply(current_patch[:, column], current_patch[:, clm])
					list_left_matrix.append(tmp)

		left_matrix = np.vstack(list_left_matrix)

		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# compute alpha
			
		list_Qjk = []
		for j in range(self.K):
			for h in range(self.K) :
				tmpQ = matmul_list([matmul_list([list_A0[j], self.UN0, self.list_Ti[h]]).T, 
					self.W, list_A[j], self.UN])
				list_Qjk.append(tmpQ)
		right_form = summatrix_list(list_Qjk)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)
		list_Pij_patch = []
		for patch_number in range(self.K):
			list_tmp = []
			for j in range(self.K):
				for h in range(self.K):
					tmpP = matmul_list([matmul_list([list_A0[j], self.UN0, self.list_Ti[h]]).T, 
						self.W, list_A0[j], self.UN0, self.list_Ti[patch_number]])
					list_tmp.append(tmpP)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pij_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pij_patch])
		self.list_alpha = np.linalg.lstsq(left_form, right_form, rcond = None)[0]
		# self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		return self.list_alpha


	def interpolate_missing(self):
		# self.normalization()

		list_U0TU = []
		for patch_number in range(self.K):
			tmp = self.list_alpha[patch_number] * matmul_list([self.UN0, self.list_Ti[patch_number], self.UN.T])
			list_U0TU.append(tmp)
		alpha_U0TU = summatrix_list(list_U0TU)

		result = np.matmul(self.A1, alpha_U0TU)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]

		# self.de_normalization()
		return result