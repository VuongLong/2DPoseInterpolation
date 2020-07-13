import numpy as np
import random
import math
from Yu_new_02.preprocess import normalize
from Yu_new_02.utils import *


class interpolation_gap_patch_PLOS_R2():
	def __init__(self, marker, missing_matrix_origin, full_test2, full_data):
		weightScale = 200
		MMweight = 0.02
		missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]

		weight_vector = compute_weight_vect_norm_v2([marker], full_data)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[marker] = MMweight
		for x in range(len(weight_vector)):
			if math.isnan(weight_vector[x]) :
				weight_vector[x] = 0
		
		missing_frame_Mzero = np.where(full_test2[:, marker*3] == 0)[0]
		list_frame = np.arange(full_test2.shape[0])
		list_full_frame_Mzero= np.asarray([i for i in list_frame if i not in missing_frame_Mzero])

		M_zero = np.copy(full_test2)

		N_nogap = np.copy(full_test2[list_full_frame_Mzero,:])
		N_zero = np.copy(N_nogap)
		N_zero[:,marker*3: marker*3+3] = 0
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

		m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
		m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		M_zero = np.multiply(((M_zero-m1) / m2),m3)

		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		N_zero = np.multiply(((N_zero-m6) / m5),m33)

		_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
		U_N_nogap = U_N_nogap_VH.T
		_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
		U_N_zero = U_N_zero_VH.T
		ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		U_N_nogap = U_N_nogap[:, :ksmall]
		U_N_zero = U_N_zero[:, :ksmall]

		self.ksmall = ksmall

		T_matrix =  np.matmul(U_N_nogap.T , U_N_zero)

		reconstruct_Mzero = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)

		m7 = np.ones((M_zero.shape[0],1))*mean_N_nogap
		m8 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		reconstruct_Mzero = m7 + (np.multiply(reconstruct_Mzero, m8) / m3)

		resultA = np.zeros(full_test2.shape)

		resultA[:, marker*3: marker*3+3] = np.copy(reconstruct_Mzero[:, marker*3: marker*3+3])
		# print(resultA[:, marker*3: marker*3+3])
		# np.savetxt("checkresult"+ str(marker) +".txt", resultA[:, marker*3: marker*3+3], fmt = "%.2f")
		# stop
		# self.reconstruct = np.copy(resultA)
		self.result = resultA



class PCA_R2():
	def __init__(self, missing_matrix):
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.copy(missing_matrix)
		self.missing_matrix = missing_matrix
		# self.reference_matrix = np.copy(reference_matrix)
		self.original_missing = missing_matrix
		self.mean_error = -1

		self.F_matrix = self.prepare()
		self.result_norm = self.interpolate_missing()

	def prepare(self, remove_patches = False, current_mean = -1):
		list_F_matrix = []
		DistalThreshold = 0.5
		test_data = np.copy(self.missing_matrix)

		AA = np.copy(self.combine_matrix)
		columnindex = np.where(AA == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		# print(markerwithgap)
		self.markerwithgap = markerwithgap
		missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
		list_frame = np.arange(test_data.shape[0])
		full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
		self.full_frame_testdata = full_frame_testdata

		[frames, columns] = AA.shape
		Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		# np.savetxt("checkMeanmat.txt", self.MeanMat[-self.missing_matrix.shape[0]:], fmt = "%.2f")
		AA = AA - self.MeanMat
		AA[np.where(self.combine_matrix == 0)] = 0
		self.norm_Data = np.copy(AA)
		resultPatch = np.zeros(self.combine_matrix.shape)

		for marker in markerwithgap:
			missing_frame = np.where(test_data[:, marker*3] == 0)
			EuclDist2Marker = compute_weight_vect_norm([marker], AA)
			thresh = np.mean(EuclDist2Marker) * DistalThreshold
			Data_remove_joint = np.copy(AA)
			for sub_marker in range(len(EuclDist2Marker)):
				if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
					Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			Data_remove_joint[:, marker*3:marker*3+3] = np.copy(AA[:, marker*3:marker*3+3])
			frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
			for sub_marker in markerwithgap:
				if sub_marker != marker:
					if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
						Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0

			gap_interpolation = interpolation_gap_patch_PLOS_R2(marker, self.missing_matrix, Data_remove_joint, np.copy(AA))

			missing_frame = np.where(Data_remove_joint[:, marker*3] == 0)[0]
			resultPatch[missing_frame, marker*3:marker*3+3] = gap_interpolation.result[missing_frame, marker*3:marker*3+3]
		# resultPatch = resultPatch[-self.missing_matrix.shape[0],:]
		return resultPatch

	def interpolate_missing(self):
		result = self.F_matrix[-self.missing_matrix.shape[0]:] + self.MeanMat[-self.missing_matrix.shape[0]:]

		final_result = np.copy(self.original_missing)
		final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
		return final_result


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


class interpolation_gap_patch():
	def __init__(self, reference_matrix, marker, missing_matrix_origin, list_frameidx_patch, full_test, full_test2):
		weightScale = 200
		MMweight = 0.02
		missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]

		weight_vector = compute_weight_vect_norm([marker], reference_matrix)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[marker] = MMweight
		for x in range(len(weight_vector)):
			if math.isnan(weight_vector[x]) :
				weight_vector[x] = 0
		self.Predic = []
		self.interpolate_A = []
		# Predic[i][j] means using Ti matrix to interpolate Aj
		N_nogap = np.copy(reference_matrix)
		N_zero = np.copy(N_nogap)
		N_zero[:,marker*3: marker*3+3] = 0
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero

		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		N_zero = np.multiply(((N_zero-m6) / m5),m33)

		M_zero = np.copy(full_test)
		m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
		m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		M_zero = np.multiply(((M_zero-m1) / m2),m3)

		# M_zero2 = np.copy(full_test2)
		# m12 = np.matmul(np.ones((M_zero2.shape[0],1)),mean_N_zero)
		# m22 = np.ones((M_zero2.shape[0],1))*stdev_N_no_gaps
		# m32 = np.matmul( np.ones((M_zero2.shape[0], 1)), column_weight)
		# M_zero2 = np.multiply(((M_zero2-m12) / m22),m32)
		
		_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
		U_N_nogap = U_N_nogap_VH.T
		_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
		U_N_zero = U_N_zero_VH.T
		ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		U_N_nogap = U_N_nogap[:, :ksmall]
		U_N_zero = U_N_zero[:, :ksmall]
		# np.savetxt("checkN_zero.txt", N_zero, fmt = "%.2f")
		# np.savetxt("checkN_nogap.txt", N_nogap, fmt = "%.2f")
		# print(ksmall)
		# stop
		self.K = int(reference_matrix.shape[0] / missing_matrix_origin.shape[0])
		for patch_number in range(self.K):
			frame_start = list_frameidx_patch[patch_number][0]
			frame_end = list_frameidx_patch[patch_number][1]
			A0 = np.copy(N_zero[frame_start: frame_end])
			A = np.copy(N_nogap[frame_start: frame_end])
			AUN = np.matmul(A, U_N_nogap)
			A0UN0 = np.matmul(A0, U_N_zero)
			
			X = np.linalg.lstsq(A0UN0, AUN, rcond = None)
			# T_matrix = np.copy(X[0])
			T_matrix = np.matmul(U_N_nogap.T, U_N_zero)
			reconstructData = np.matmul(np.matmul(np.matmul(N_zero, U_N_zero), T_matrix), U_N_nogap.T)

			# reverse normalization
			m7 = np.ones((reference_matrix.shape[0],1))*mean_N_nogap
			m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
			m3 = np.matmul( np.ones((reference_matrix.shape[0], 1)), column_weight)
			reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)

			list_prec = []
			for i in range(self.K):
				Fs = list_frameidx_patch[i][0]
				Fe = list_frameidx_patch[i][1]
				Ai = np.copy(reconstructData[Fs:Fe])
				resultAi = np.zeros(missing_matrix_origin.shape)
				resultAi[missing_frame, marker*3: marker*3+3] = Ai[missing_frame, marker*3: marker*3+3]
				list_prec.append(resultAi)

			self.Predic.append(list_prec)

			reconstruct_Mzero = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
			# reconstruct_Mzero2 = np.matmul(np.matmul(np.matmul(M_zero2, U_N_zero), T_matrix), U_N_nogap.T)
			
			# print(np.sum(np.abs(reconstruct_Mzero2)) - np.sum(np.abs(reconstruct_Mzero)))
			# stop
			m7 = np.ones((M_zero.shape[0],1))*mean_N_nogap
			m8 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
			m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
			reconstruct_Mzero = m7 + (np.multiply(reconstruct_Mzero, m8) / m3)
			# reconstruct_Mzero2 = m7 + (np.multiply(reconstruct_Mzero2, m8) / m3)
			# self.reconstruct = reconstruct_Mzero2

			number_missing = len(missing_frame)
			# predicA = np.copy(reconstruct_Mzero[-number_missing:])
			resultA = np.zeros(missing_matrix_origin.shape)
			predicA = np.copy(reconstruct_Mzero[-number_missing:])
			# resultA = np.copy(missing_matrix_origin)
			resultA[missing_frame, marker*3: marker*3+3] = np.copy(predicA[:, marker*3: marker*3+3])
			# np.savetxt("checkresult.txt", resultA, fmt = "%.2f")
			# stop
			# self.reconstruct = np.copy(resultA)
			self.interpolate_A.append(resultA)


class interpolation_gap_patch_v2():
	def __init__(self, reference_matrix, marker, missing_matrix_origin, list_frameidx_patch, full_test, full_test2, full_data):
		weightScale = 200
		MMweight = 0.02
		missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]

		weight_vector = compute_weight_vect_norm_v2([marker], full_data)
		# print("return weight_vector")
		# print(weight_vector)
		# print("end")
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[marker] = MMweight
		for x in range(len(weight_vector)):
			if math.isnan(weight_vector[x]) :
				weight_vector[x] = 0
		self.Predic = []
		self.Predic_f = []
		self.interpolate_A = []
		# Predic[i][j] means using Ti matrix to interpolate Aj
		missing_frame_Mzero = np.where(full_test2[:, marker*3] == 0)[0]
		list_frame = np.arange(full_test2.shape[0])
		list_full_frame_Mzero= np.asarray([i for i in list_frame if i not in missing_frame_Mzero])
		
		M_zero = np.copy(full_test2)
		
		N_nogap = np.copy(full_test2[list_full_frame_Mzero,:])
		# print("debugggggg")
		# print(N_nogap.shape)
		# print("end/////////////////////////////////////////////")
		N_zero = np.copy(N_nogap)
		N_zero[:,marker*3: marker*3+3] = 0
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero
		
		m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
		m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		M_zero = np.multiply(((M_zero-m1) / m2),m3)

		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		N_zero = np.multiply(((N_zero-m6) / m5),m33)

		
		# M_zero2 = np.copy(full_test2)
		# m12 = np.matmul(np.ones((M_zero2.shape[0],1)),mean_N_zero)
		# m22 = np.ones((M_zero2.shape[0],1))*stdev_N_no_gaps
		# m32 = np.matmul( np.ones((M_zero2.shape[0], 1)), column_weight)
		# M_zero2 = np.multiply(((M_zero2-m12) / m22),m32)
		
		_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
		U_N_nogap = U_N_nogap_VH.T
		_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
		U_N_zero = U_N_zero_VH.T
		ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		U_N_nogap = U_N_nogap[:, :ksmall]
		U_N_zero = U_N_zero[:, :ksmall]
		# np.savetxt("checkN_zero.txt", N_zero, fmt = "%.2f")
		# np.savetxt("checkN_nogap.txt", N_nogap, fmt = "%.2f")
		# print(ksmall)
		# stop
		self.ksmall = ksmall
		self.K = int(reference_matrix.shape[0] / missing_matrix_origin.shape[0])
		for patch_number in range(self.K):
			frame_start = list_frameidx_patch[patch_number][0]
			frame_end = list_frameidx_patch[patch_number][1]
			A0 = np.copy(N_zero[frame_start: frame_end])
			A = np.copy(N_nogap[frame_start: frame_end])
			AUN = np.matmul(A, U_N_nogap)
			A0UN0 = np.matmul(A0, U_N_zero)
			
			X = np.linalg.lstsq(A0UN0, AUN, rcond = None)
			T_matrix = np.copy(X[0])
			# T_matrix = np.matmul(U_N_nogap.T, U_N_zero)
			# np.savetxt("checkT" + str(patch_number) + "after.txt", T_matrix, fmt = "%.2f")
			# print("enter")
			# print(patch_number)
			reconstructData = np.matmul(np.matmul(np.matmul(N_zero, U_N_zero), T_matrix), U_N_nogap.T)

			# reverse normalization
			m7 = np.ones((reconstructData.shape[0],1))*mean_N_nogap
			m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
			m3 = np.matmul( np.ones((reconstructData.shape[0], 1)), column_weight)
			reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)

			list_prec = []
			list_prec_f = []
			for i in range(self.K):
				Fs = list_frameidx_patch[i][0]
				Fe = list_frameidx_patch[i][1]
				Ai = np.copy(reconstructData[Fs:Fe])
				resultAi = np.zeros(missing_matrix_origin.shape)
				resultAi[missing_frame, marker*3: marker*3+3] = Ai[missing_frame, marker*3: marker*3+3]
				list_prec.append(resultAi)
				list_prec_f.append(Ai)

			self.Predic.append(list_prec)
			self.Predic_f.append(list_prec_f)

			reconstruct_Mzero = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
			# reconstruct_Mzero2 = np.matmul(np.matmul(np.matmul(M_zero2, U_N_zero), T_matrix), U_N_nogap.T)
			# print(np.sum(np.abs(reconstruct_Mzero2)) - np.sum(np.abs(reconstruct_Mzero)))
			# stop
			m7 = np.ones((M_zero.shape[0],1))*mean_N_nogap
			m8 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
			m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
			reconstruct_Mzero = m7 + (np.multiply(reconstruct_Mzero, m8) / m3)
			# reconstruct_Mzero2 = m7 + (np.multiply(reconstruct_Mzero2, m8) / m3)
			# self.reconstruct = reconstruct_Mzero2

			# number_missing = len(missing_frame)
			# predicA = np.copy(reconstruct_Mzero[-number_missing:])
			resultA = np.zeros(full_test2.shape)
			# resultA = np.copy(missing_matrix_origin)
			resultA[:, marker*3: marker*3+3] = np.copy(reconstruct_Mzero[:, marker*3: marker*3+3])
			# print(resultA[:, marker*3: marker*3+3])
			# np.savetxt("checkresult"+ str(marker) +".txt", resultA[:, marker*3: marker*3+3], fmt = "%.2f")
			# stop
			# self.reconstruct = np.copy(resultA)
			self.interpolate_A.append(resultA)

def compute_error_patch(fullData, reconstructData, marker, missing_matrix_origin):
	[xx, yy] = missing_matrix_origin.shape
	data = fullData[-xx:]
	compareData = reconstructData[-xx:]
	missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]
	fullFrame = [x for x in range(xx) if x not in missing_frame]
	result = np.sum(np.abs(data[fullFrame, marker*3: marker*3+3] - compareData[fullFrame, marker*3: marker*3+3]))
	number_maker = xx - len(missing_frame)
	return result / (number_maker * 3)

class interpolation_gap_patch_v3():
	def __init__(self, reference_matrix, marker, missing_matrix_origin, list_frameidx_patch, full_test, full_test2, full_data):
		weightScale = 200
		MMweight = 0.02
		missing_frame = np.where(missing_matrix_origin[:, marker*3] == 0)[0]

		weight_vector = compute_weight_vect_norm_v2([marker], full_data)
		# print("return weight_vector")
		# print(weight_vector)
		# print("end")
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[marker] = MMweight
		for x in range(len(weight_vector)):
			if math.isnan(weight_vector[x]) :
				weight_vector[x] = 0
		self.Predic = []
		self.Predic_f = []
		self.interpolate_A = []
		self.list_error_patch = []
		# Predic[i][j] means using Ti matrix to interpolate Aj
		missing_frame_Mzero = np.where(full_test2[:, marker*3] == 0)[0]
		list_frame = np.arange(full_test2.shape[0])
		list_full_frame_Mzero= np.asarray([i for i in list_frame if i not in missing_frame_Mzero])
		M_zero = np.copy(full_test2)
		
		N_nogap = np.copy(full_test2[list_full_frame_Mzero,:])
		# print("debugggggg")
		# print(N_nogap.shape)
		# print("end/////////////////////////////////////////////")
		N_zero = np.copy(N_nogap)
		N_zero[:,marker*3: marker*3+3] = 0
		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		mean_N_zero = mean_N_zero.reshape((1, mean_N_zero.shape[0]))
		stdev_N_no_gaps = np.std(N_nogap, 0)
		stdev_N_no_gaps[np.where(stdev_N_no_gaps == 0)] = 1


		column_weight = np.ravel(np.ones((3,1)) * weight_vector, order='F')
		column_weight = column_weight.reshape((1, column_weight.shape[0]))
		m33 = np.matmul( np.ones((N_nogap.shape[0], 1)), column_weight)
		m4 = np.ones((N_nogap.shape[0],1))*mean_N_nogap
		m5 = np.ones((N_nogap.shape[0],1))*stdev_N_no_gaps
		m6 = np.ones((N_zero.shape[0],1))*mean_N_zero
		
		m1 = np.matmul(np.ones((M_zero.shape[0],1)),mean_N_zero)
		m2 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
		m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
		M_zero = np.multiply(((M_zero-m1) / m2),m3)

		N_nogap = np.multiply(((N_nogap-m4)/ m5),m33)
		N_zero = np.multiply(((N_zero-m6) / m5),m33)

		
		# M_zero2 = np.copy(full_test2)
		# m12 = np.matmul(np.ones((M_zero2.shape[0],1)),mean_N_zero)
		# m22 = np.ones((M_zero2.shape[0],1))*stdev_N_no_gaps
		# m32 = np.matmul( np.ones((M_zero2.shape[0], 1)), column_weight)
		# M_zero2 = np.multiply(((M_zero2-m12) / m22),m32)
		
		_, Sigma_nogap , U_N_nogap_VH = np.linalg.svd(N_nogap/np.sqrt(N_nogap.shape[0]-1), full_matrices = False)
		U_N_nogap = U_N_nogap_VH.T
		_, Sigma_zero , U_N_zero_VH = np.linalg.svd(N_zero/np.sqrt(N_zero.shape[0]-1), full_matrices = False)
		U_N_zero = U_N_zero_VH.T
		ksmall = max(setting_rank(Sigma_zero), setting_rank(Sigma_nogap))
		U_N_nogap = U_N_nogap[:, :ksmall]
		U_N_zero = U_N_zero[:, :ksmall]
		# np.savetxt("checkN_zero.txt", N_zero, fmt = "%.2f")
		# np.savetxt("checkN_nogap.txt", N_nogap, fmt = "%.2f")
		# print(ksmall)
		# stop
		self.ksmall = ksmall
		self.K = int(reference_matrix.shape[0] / missing_matrix_origin.shape[0])
		for patch_number in range(self.K):
			frame_start = list_frameidx_patch[patch_number][0]
			frame_end = list_frameidx_patch[patch_number][1]
			A0 = np.copy(N_zero[frame_start: frame_end])
			A = np.copy(N_nogap[frame_start: frame_end])
			AUN = np.matmul(A, U_N_nogap)
			A0UN0 = np.matmul(A0, U_N_zero)
			
			X = np.linalg.lstsq(A0UN0, AUN, rcond = None)
			T_matrix = np.copy(X[0])
			# T_matrix = np.matmul(U_N_nogap.T, U_N_zero)
			# np.savetxt("checkT" + str(patch_number) + ".txt", T_matrix, fmt = "%.2f")
			reconstructData = np.matmul(np.matmul(np.matmul(N_zero, U_N_zero), T_matrix), U_N_nogap.T)

			# reverse normalization
			m7 = np.ones((reconstructData.shape[0],1))*mean_N_nogap
			m8 = np.ones((reconstructData.shape[0],1))*stdev_N_no_gaps
			m3 = np.matmul( np.ones((reconstructData.shape[0], 1)), column_weight)
			reconstructData = m7 + (np.multiply(reconstructData, m8) / m3)

			list_prec = []
			list_prec_f = []
			for i in range(self.K):
				Fs = list_frameidx_patch[i][0]
				Fe = list_frameidx_patch[i][1]
				Ai = np.copy(reconstructData[Fs:Fe])
				resultAi = np.zeros(missing_matrix_origin.shape)
				resultAi[missing_frame, marker*3: marker*3+3] = Ai[missing_frame, marker*3: marker*3+3]
				list_prec.append(resultAi)
				list_prec_f.append(Ai)

			self.Predic.append(list_prec)
			self.Predic_f.append(list_prec_f)

			reconstruct_Mzero = np.matmul(np.matmul(np.matmul(M_zero, U_N_zero), T_matrix), U_N_nogap.T)
			# reconstruct_Mzero2 = np.matmul(np.matmul(np.matmul(M_zero2, U_N_zero), T_matrix), U_N_nogap.T)
			# print(np.sum(np.abs(reconstruct_Mzero2)) - np.sum(np.abs(reconstruct_Mzero)))
			# stop
			m7 = np.ones((M_zero.shape[0],1))*mean_N_nogap
			m8 = np.ones((M_zero.shape[0],1))*stdev_N_no_gaps
			m3 = np.matmul( np.ones((M_zero.shape[0], 1)), column_weight)
			reconstruct_Mzero = m7 + (np.multiply(reconstruct_Mzero, m8) / m3)
			# reconstruct_Mzero2 = m7 + (np.multiply(reconstruct_Mzero2, m8) / m3)
			# self.reconstruct = reconstruct_Mzero2

			# number_missing = len(missing_frame)
			# predicA = np.copy(reconstruct_Mzero[-number_missing:])
			resultA = np.zeros(full_test2.shape)
			# resultA = np.copy(missing_matrix_origin)
			resultA[:, marker*3: marker*3+3] = np.copy(reconstruct_Mzero[:, marker*3: marker*3+3])
			error = compute_error_patch(full_test2, resultA, marker, missing_matrix_origin)
			self.list_error_patch.append(error)
			# print(resultA[:, marker*3: marker*3+3])
			# np.savetxt("checkresult"+ str(marker) +".txt", resultA[:, marker*3: marker*3+3], fmt = "%.2f")
			# stop
			# self.reconstruct = np.copy(resultA)
			self.interpolate_A.append(resultA)


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
			
			X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
			self.list_Ti.append(np.copy(X[0]))

			# self.list_Ti.append(np.matmul(self.UN.T, self.UN0))
			
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
		print(self.list_alpha)
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


class interpolation_weighted_gap():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries, self.list_N0_gap = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		self.compute_svd()
		self.result_norm = self.interpolate_missing()
		self.de_normalization()

	def normalization(self):
		AA = np.copy(self.combine_matrix)
		weightScale = 200
		MMweight = 0.02
		[frames, columns] = AA.shape
		columnindex = np.where(AA == 0)[1]
		frameindex = np.where(AA == 0)[0]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		self.markerwithgap = markerwithgap
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
		downsample = True
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
		weight_vector = np.min(weight_vector, 0)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		M_zero = np.copy(Data)
		# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
		N_nogap = np.delete(Data, framewithgap, 0)
		N_nogap_origin = np.copy(N_nogap)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0

		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		tmp_tmp = np.copy(mean_N_zero)
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
		list_N0_gap = []
		# looping for gaps
		for marker in markerwithgap:
			N_zero_tmp = np.copy(N_nogap_origin)
			N_zero_tmp[:,marker*3 : marker*3+3] = 0

			mean_N_zero_tmp = np.mean(N_zero_tmp, 0)
			mean_N_zero_tmp = mean_N_zero_tmp.reshape((1, mean_N_zero_tmp.shape[0]))
			m6 = np.ones((N_zero_tmp.shape[0],1))*mean_N_zero_tmp

			N_zero_tmp = np.multiply(((N_zero_tmp-m6) / m5),m33)
			list_N0_gap.append(N_zero_tmp)
		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat], list_N0_gap


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
		
		list_left_matrix_G = []
		self.list_Ti_gap = []
		self.list_UN0_gap = []
		
		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)
		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma)) - 1
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		counter_gap = 0
		
		# divide original A0 per patch
		list_A0_origin = []
		list_A_origin = []
		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A0_origin.append(np.copy(p_AN0[l:r]))
			list_A_origin.append(np.copy(p_AN[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		# end section
		Q_missing_byPatch = np.zeros(list_A0_origin[0].shape)
		for marker in self.markerwithgap:
			Q_missing_byPatch[:, marker*3: marker*3+3] = 1
		Q_deduct_missing = np.ones(list_A0_origin[0].shape)
		Q_deduct_missing[np.where(Q_missing_byPatch == 1)] = 0
		list_Q_gap = []	
		# prepare list_Q_gap 

		for marker in self.markerwithgap:
			tmp = np.zeros(list_A0_origin[0].shape)
			tmp[:, marker*3 : marker*3+3] = 1
			list_Q_gap.append(np.copy(tmp))
		# end
		self.Q_gap = []
		for marker in self.markerwithgap:
			tmp = np.zeros(self.A1.shape)
			tmp[:, marker*3 : marker*3+3] = 1
			self.Q_gap.append(tmp)

		# divide A0 per patch that regard to missing gaps

		list_A0_gaps = []
		# list_A0_gaps contains all the A0 that miss gap respectively
		for marker in self.markerwithgap:
			list_A0 = []
			#  list_A0 contains the A0 missing only gap marker
			r = len(self.AN0)
			l = r - self.fix_leng
			
			AN0_gapG = self.list_N0_gap[counter_gap]
			_, tmp_U0sigma, tmp_U0 = np.linalg.svd(AN0_gapG/np.sqrt(AN0_gapG.shape[0]-1), full_matrices = False)
			current_UN0 = np.copy(tmp_U0.T)[:, :ksmall]
			self.list_UN0_gap.append(current_UN0)
			# compute UN0 corresponds to the current missing marker g

			while l >= 0:
				list_A0.append(np.copy(AN0_gapG[l:r]))
				l -= self.fix_leng
				r -= self.fix_leng

			list_A0_gaps.append(list_A0)
		
			list_Ti = []

			for patch_number in range(self.K):
				AiUN = np.matmul(list_A_origin[patch_number], self.UN)
				Ai0UN0 = np.matmul(list_A0[patch_number], current_UN0)
				
				X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
				list_Ti.append(np.copy(X[0]))

				# list_Ti.append(np.matmul(self.UN.T, current_UN0))
			
			# compute Ti corresponds to current missing marker g
				
			self.list_Ti_gap.append(list_Ti)		
	
			list_P = []
			for patch_number in range(self.K):
				tmp = list_A_origin[patch_number] - np.matmul(list_A0[patch_number], 
						np.matmul(np.matmul(current_UN0, list_Ti[patch_number]), self.UN.T))
				list_P.append(tmp)
			P_matrix = summatrix_list(list_P)
			P_matrix = P_matrix * list_Q_gap[counter_gap]

			joint_number = P_matrix.shape[1]
			list_P_column = []
			for i in range(joint_number):
				left_P = P_matrix[:,i]
				for j in range(joint_number):
					right_P = P_matrix[:,j]
					list_P_column.append(np.copy(left_P * right_P))
			left_matrix_tmp = np.vstack(list_P_column)
			list_left_matrix_G.append(left_matrix_tmp)
			counter_gap	+= 1 # increase index in list_N0_gap and list_Q_gap
		# end section
		# compute weight	

		left_matrix = summatrix_list(list_left_matrix_G)
		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# end
		# compute alpha
		

		list_Qgj = []
		for g in range(len(self.markerwithgap)):
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g], current_Ti[j]), self.UN.T])
				tmp_left = tmp_left * list_Q_gap[g]
				tmp_left = tmp_left + current_A0[j]
				tmpQ =  matmul_list([tmp_left.T, self.W, list_A_origin[j]])
				list_Qgj.append(tmpQ)
		right_form = summatrix_list(list_Qgj)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		
		list_Pgj_patch = []
		for alpha_g in range(len(self.markerwithgap)):
			list_tmp = []
			alpha_Ti = self.list_Ti_gap[alpha_g]
			for g in range(len(self.markerwithgap)):
				current_A0 = list_A0_origin
				current_Ti = self.list_Ti_gap[g]
				for j in range(self.K):
					tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g],current_Ti[j]), self.UN.T])
					tmp_left = tmp_left * list_Q_gap[g]
					tmp_left = tmp_left + current_A0[j]
					tmp_right = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[alpha_g], alpha_Ti[j]), self.UN.T ])
					tmp_right = tmp_right * list_Q_gap[alpha_g]
					tmp_right = tmp_right + current_A0[j]
					tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])						
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pgj_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pgj_patch])
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print(self.list_alpha)
		# end
		# compute beta

		Q_missing_byPatch = np.ones(Q_missing_byPatch.shape)

		list_Pj = []
		for j in range(self.K):
			list_tmp = []
			for g in range(len(self.markerwithgap)):
				tmp = self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_tmp.append(np.copy(tmp))
			list_Pj.append(summatrix_list(list_tmp))

		list_Zik = []
		for beta_loop in range(self.K):
			for i in range(self.K):
				# compute P betaloop inside
				tmp_left = np.matmul(list_Pj[beta_loop].T, list_A0_origin[i].T) * Q_missing_byPatch.T
				tmp = matmul_list([tmp_left, self.W, list_A_origin[i]])
				list_Zik.append(np.copy(tmp))
		right_form = summatrix_list(list_Zik)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		list_Yik_beta = []
		# betaindex : beta1, beta2, .. k beta
		# betaloop : beta1 1st time, 2nd time ... k times
		for beta_index in range(self.K):
			list_tmp = []
			for beta_loop in range(self.K):
				for i in range(self.K):
					# compute P betaloop
					tmp_left = np.matmul(list_Pj[beta_loop].T,list_A0_origin[i].T) * Q_missing_byPatch.T
					# compute P beta index
					tmp_right = np.matmul(list_A0_origin[i], list_Pj[beta_index]) * Q_missing_byPatch

					tmp_P = matmul_list( [tmp_left, self.W, tmp_right])	
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Yik_beta.append(tmp.reshape(xx*yy, 1))
		left_form = np.hstack([ x for x in list_Yik_beta])
		self.list_beta = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print("list beta")
		print(self.list_beta)
		return 0


	def interpolate_missing(self):
		# self.normalization()
		list_matrix = []
		for j in range(self.K):
			for g in range(len(self.markerwithgap)):
				tmp = self.list_beta[j] * self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_matrix.append(np.matmul(self.A1, tmp) * self.Q_gap[g])
		result = summatrix_list(list_matrix)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]
		# self.de_normalization()
		return final_result



class interpolation_weighted_gap_Yu_v2():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries, self.list_N0_gap = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		self.compute_svd()
		self.result_norm = self.interpolate_missing()
		self.de_normalization()

	def normalization(self):
		AA = np.copy(self.combine_matrix)
		weightScale = 200
		MMweight = 0.02
		[frames, columns] = AA.shape
		columnindex = np.where(AA == 0)[1]
		frameindex = np.where(AA == 0)[0]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		self.markerwithgap = markerwithgap
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
		downsample = True
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
		weight_vector = np.min(weight_vector, 0)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		M_zero = np.copy(Data)
		# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
		N_nogap = np.delete(Data, framewithgap, 0)
		N_nogap_origin = np.copy(N_nogap)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0

		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		tmp_tmp = np.copy(mean_N_zero)
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
		list_N0_gap = []
		# looping for gaps
		for marker in markerwithgap:
			N_zero_tmp = np.copy(N_nogap_origin)
			N_zero_tmp[:,marker*3 : marker*3+3] = 0

			mean_N_zero_tmp = np.mean(N_zero_tmp, 0)
			mean_N_zero_tmp = mean_N_zero_tmp.reshape((1, mean_N_zero_tmp.shape[0]))
			m6 = np.ones((N_zero_tmp.shape[0],1))*mean_N_zero_tmp

			N_zero_tmp = np.multiply(((N_zero_tmp-m6) / m5),m33)
			list_N0_gap.append(N_zero_tmp)
		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat], list_N0_gap


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
		
		list_left_matrix_G = []
		self.list_Ti_gap = []
		self.list_UN0_gap = []
		
		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)
		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma)) - 1
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		counter_gap = 0
		
		# divide original A0 per patch
		list_A0_origin = []
		list_A_origin = []
		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A0_origin.append(np.copy(p_AN0[l:r]))
			list_A_origin.append(np.copy(p_AN[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		# end section
		Q_missing_byPatch = np.zeros(list_A0_origin[0].shape)
		for marker in self.markerwithgap:
			Q_missing_byPatch[:, marker*3: marker*3+3] = 1
		Q_deduct_missing = np.ones(list_A0_origin[0].shape)
		Q_deduct_missing[np.where(Q_missing_byPatch == 1)] = 0
		list_Q_gap = []	
		# prepare list_Q_gap 

		for marker in self.markerwithgap:
			tmp = np.zeros(list_A0_origin[0].shape)
			tmp[:, marker*3 : marker*3+3] = 1
			list_Q_gap.append(np.copy(tmp))
		# end
		self.Q_gap = []
		for marker in self.markerwithgap:
			tmp = np.zeros(self.A1.shape)
			tmp[:, marker*3 : marker*3+3] = 1
			self.Q_gap.append(tmp)

		# divide A0 per patch that regard to missing gaps

		list_A0_gaps = []
		# list_A0_gaps contains all the A0 that miss gap respectively
		for marker in self.markerwithgap:
			list_A0 = []
			#  list_A0 contains the A0 missing only gap marker
			r = len(self.AN0)
			l = r - self.fix_leng
			
			AN0_gapG = self.list_N0_gap[counter_gap]
			_, tmp_U0sigma, tmp_U0 = np.linalg.svd(AN0_gapG/np.sqrt(AN0_gapG.shape[0]-1), full_matrices = False)
			current_UN0 = np.copy(tmp_U0.T)[:, :ksmall]
			self.list_UN0_gap.append(current_UN0)
			# compute UN0 corresponds to the current missing marker g

			while l >= 0:
				list_A0.append(np.copy(AN0_gapG[l:r]))
				l -= self.fix_leng
				r -= self.fix_leng

			list_A0_gaps.append(list_A0)
		
			list_Ti = []

			for patch_number in range(self.K):
				AiUN = np.matmul(list_A_origin[patch_number], self.UN)
				Ai0UN0 = np.matmul(list_A0[patch_number], current_UN0)
				
				X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
				list_Ti.append(np.copy(X[0]))

				# list_Ti.append(np.matmul(self.UN.T, current_UN0))
			
			# compute Ti corresponds to current missing marker g
				
			self.list_Ti_gap.append(list_Ti)		
	
			list_P = []
			for patch_number in range(self.K):
				tmp = list_A_origin[patch_number] - np.matmul(list_A0[patch_number], 
						np.matmul(np.matmul(current_UN0, list_Ti[patch_number]), self.UN.T))
				list_P.append(tmp)
			P_matrix = summatrix_list(list_P)
			P_matrix = P_matrix * list_Q_gap[counter_gap]

			joint_number = P_matrix.shape[1]
			list_P_column = []
			for i in range(joint_number):
				left_P = P_matrix[:,i]
				for j in range(joint_number):
					right_P = P_matrix[:,j]
					list_P_column.append(np.copy(left_P * right_P))
			left_matrix_tmp = np.vstack(list_P_column)
			list_left_matrix_G.append(left_matrix_tmp)
			counter_gap	+= 1 # increase index in list_N0_gap and list_Q_gap
		# end section
		# compute weight	

		left_matrix = summatrix_list(list_left_matrix_G)
		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# end
		# compute alpha
		
		Q_missing_byPatch = np.ones(Q_missing_byPatch.shape)


		# # proving formula:

		# print(self.K)
		# # list_A_origin[0] 
		# # A .* Q = A_0^(0) (sum alpha * (U_0 T_j)_g U^T ) .* Q
		
		# list_matrix = []
		# for g in range(len(self.markerwithgap)):
		# 	current_Ti = self.list_Ti_gap[g]
		# 	tmp_matrix = np.matmul(np.matmul(self.list_UN0_gap[g], current_Ti[0]),self.UN.T)
		# 	list_matrix.append(tmp_matrix)
		# U0TU = summatrix_list(list_matrix)
		# result = np.matmul(list_A0_origin[0], U0TU) * Q_missing_byPatch
		# origin = list_A_origin[0] * Q_missing_byPatch
		# compare_matrix = np.abs(result - origin)
		# print(np.sum(compare_matrix))
		# list_matrix = []
		# for g in range(len(self.markerwithgap)):
		# 	current_Ti = self.list_Ti_gap[g]
		# 	tmp_matrix = np.matmul(list_A0_origin[0], np.matmul(np.matmul(self.list_UN0_gap[g], current_Ti[0]),self.UN.T)) * list_Q_gap[g]
		# 	list_matrix.append(tmp_matrix)
		# result = summatrix_list(list_matrix)
		# origin = list_A_origin[0] * Q_missing_byPatch
		# compare_matrix = np.abs(result - origin)
		# print(np.sum(compare_matrix))
		# stop


		# end proving
		list_Qgj = []
		for g in range(len(self.markerwithgap)):
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g], current_Ti[j]), self.UN.T])
				tmp_left = tmp_left * list_Q_gap[g]
				tmp_left = tmp_left + current_A0[j]
				tmpQ =  matmul_list([tmp_left.T, self.W, list_A_origin[j]])
				list_Qgj.append(tmpQ)
		right_form = summatrix_list(list_Qgj)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		
		list_Pgj_patch = []
		for alpha_g in range(len(self.markerwithgap)):
			list_tmp = []
			alpha_Ti = self.list_Ti_gap[alpha_g]
			for g in range(len(self.markerwithgap)):
				current_A0 = list_A0_origin
				current_Ti = self.list_Ti_gap[g]
				for j in range(self.K):
					tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g],current_Ti[j]), self.UN.T])
					tmp_left = tmp_left * list_Q_gap[g]
					tmp_left = tmp_left + current_A0[j]
					tmp_right = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[alpha_g], alpha_Ti[j]), self.UN.T ])
					tmp_right = tmp_right * list_Q_gap[alpha_g]
					tmp_right = tmp_right + current_A0[j]
					tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])						
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pgj_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pgj_patch])
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print(self.list_alpha)
		# end
		# compute beta


		list_Pj = []
		for j in range(self.K):
			list_tmp = []
			for g in range(len(self.markerwithgap)):
				tmp = self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_tmp.append(np.copy(tmp))
			list_Pj.append(summatrix_list(list_tmp))

		list_Zik = []
		for beta_loop in range(self.K):
			for i in range(self.K):
				# compute P betaloop inside
				tmp_left = np.matmul(list_Pj[beta_loop].T, list_A0_origin[i].T) * Q_missing_byPatch.T
				tmp = matmul_list([tmp_left, self.W, list_A_origin[i]])
				list_Zik.append(np.copy(tmp))
		right_form = summatrix_list(list_Zik)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		list_Yik_beta = []
		# betaindex : beta1, beta2, .. k beta
		# betaloop : beta1 1st time, 2nd time ... k times
		for beta_index in range(self.K):
			list_tmp = []
			for beta_loop in range(self.K):
				for i in range(self.K):
					# compute P betaloop
					tmp_left = np.matmul(list_Pj[beta_loop].T,list_A0_origin[i].T) * Q_missing_byPatch.T
					# compute P beta index
					tmp_right = np.matmul(list_A0_origin[i], list_Pj[beta_index]) * Q_missing_byPatch

					tmp_P = matmul_list( [tmp_left, self.W, tmp_right])	
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Yik_beta.append(tmp.reshape(xx*yy, 1))
		left_form = np.hstack([ x for x in list_Yik_beta])
		self.list_beta = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print("list beta")
		print(self.list_beta)
		return 0


	def interpolate_missing(self):
		# self.normalization()
		list_matrix = []
		for j in range(self.K):
			for g in range(len(self.markerwithgap)):
				tmp = self.list_beta[j] * self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_matrix.append(np.matmul(self.A1, tmp) * self.Q_gap[g])
		result = summatrix_list(list_matrix)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]
		# self.de_normalization()
		return final_result



class interpolation_weighted_gap_Yu_v3():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries, self.list_N0_gap = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		self.compute_svd()
		self.result_norm = self.interpolate_missing()
		self.de_normalization()

	def normalization(self):
		AA = np.copy(self.combine_matrix)
		weightScale = 200
		MMweight = 0.02
		[frames, columns] = AA.shape
		columnindex = np.where(AA == 0)[1]
		frameindex = np.where(AA == 0)[0]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		self.markerwithgap = markerwithgap
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
		downsample = True
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
		weight_vector = np.min(weight_vector, 0)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		M_zero = np.copy(Data)
		# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
		N_nogap = np.delete(Data, framewithgap, 0)
		N_nogap_origin = np.copy(N_nogap)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0

		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		tmp_tmp = np.copy(mean_N_zero)
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
		list_N0_gap = []
		# looping for gaps
		for marker in markerwithgap:
			N_zero_tmp = np.copy(N_nogap_origin)
			N_zero_tmp[:,marker*3 : marker*3+3] = 0

			mean_N_zero_tmp = np.mean(N_zero_tmp, 0)
			mean_N_zero_tmp = mean_N_zero_tmp.reshape((1, mean_N_zero_tmp.shape[0]))
			m6 = np.ones((N_zero_tmp.shape[0],1))*mean_N_zero_tmp

			N_zero_tmp = np.multiply(((N_zero_tmp-m6) / m5),m33)
			list_N0_gap.append(N_zero_tmp)
		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat], list_N0_gap


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
		
		list_left_matrix_G = []
		self.list_Ti_gap = []
		self.list_UN0_gap = []
		
		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)
		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma)) - 1
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		counter_gap = 0
		
		# divide original A0 per patch
		list_A0_origin = []
		list_A_origin = []
		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A0_origin.append(np.copy(p_AN0[l:r]))
			list_A_origin.append(np.copy(p_AN[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		# end section
		Q_missing_byPatch = np.zeros(list_A0_origin[0].shape)
		for marker in self.markerwithgap:
			Q_missing_byPatch[:, marker*3: marker*3+3] = 1
		Q_deduct_missing = np.ones(list_A0_origin[0].shape)
		Q_deduct_missing[np.where(Q_missing_byPatch == 1)] = 0
		list_Q_gap = []	
		# prepare list_Q_gap 

		for marker in self.markerwithgap:
			tmp = np.zeros(list_A0_origin[0].shape)
			tmp[:, marker*3 : marker*3+3] = 1
			list_Q_gap.append(np.copy(tmp))
		# end
		self.Q_gap = []
		for marker in self.markerwithgap:
			tmp = np.zeros(self.A1.shape)
			tmp[:, marker*3 : marker*3+3] = 1
			self.Q_gap.append(tmp)

		# divide A0 per patch that regard to missing gaps

		list_A0_gaps = []
		# list_A0_gaps contains all the A0 that miss gap respectively
		for marker in self.markerwithgap:
			list_A0 = []
			#  list_A0 contains the A0 missing only gap marker
			r = len(self.AN0)
			l = r - self.fix_leng
			
			AN0_gapG = self.list_N0_gap[counter_gap]
			_, tmp_U0sigma, tmp_U0 = np.linalg.svd(AN0_gapG/np.sqrt(AN0_gapG.shape[0]-1), full_matrices = False)
			current_UN0 = np.copy(tmp_U0.T)[:, :ksmall]
			self.list_UN0_gap.append(current_UN0)
			# compute UN0 corresponds to the current missing marker g

			while l >= 0:
				list_A0.append(np.copy(AN0_gapG[l:r]))
				l -= self.fix_leng
				r -= self.fix_leng

			list_A0_gaps.append(list_A0)
		
			list_Ti = []

			for patch_number in range(self.K):
				AiUN = np.matmul(list_A_origin[patch_number], self.UN)
				Ai0UN0 = np.matmul(list_A0[patch_number], current_UN0)
				
				X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
				list_Ti.append(np.copy(X[0]))

				# list_Ti.append(np.matmul(self.UN.T, current_UN0))
			
			# compute Ti corresponds to current missing marker g
				
			self.list_Ti_gap.append(list_Ti)		
	
			list_P = []
			for patch_number in range(self.K):
				tmp = list_A_origin[patch_number] - np.matmul(list_A0[patch_number], 
						np.matmul(np.matmul(current_UN0, list_Ti[patch_number]), self.UN.T))
				list_P.append(tmp)
			P_matrix = summatrix_list(list_P)
			P_matrix = P_matrix * list_Q_gap[counter_gap]

			joint_number = P_matrix.shape[1]
			list_P_column = []
			for i in range(joint_number):
				left_P = P_matrix[:,i]
				for j in range(joint_number):
					right_P = P_matrix[:,j]
					list_P_column.append(np.copy(left_P * right_P))
			left_matrix_tmp = np.vstack(list_P_column)
			list_left_matrix_G.append(left_matrix_tmp)
			counter_gap	+= 1 # increase index in list_N0_gap and list_Q_gap
		# end section
		# compute weight	

		left_matrix = summatrix_list(list_left_matrix_G)
		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# end
		# compute alpha
		
		Q_missing_byPatch = np.ones(Q_missing_byPatch.shape)

		stop
		list_Qgj = []
		for g in range(len(self.markerwithgap)):
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g], current_Ti[j]), self.UN.T])
				tmp_left = tmp_left * list_Q_gap[g]
				tmp_left = tmp_left + current_A0[j]
				tmpQ =  matmul_list([tmp_left.T, self.W, list_A_origin[j]])
				list_Qgj.append(tmpQ)
		right_form = summatrix_list(list_Qgj)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		stop
		list_Pgj_patch = []
		for alpha_g in range(len(self.markerwithgap)):
			list_tmp = []
			alpha_Ti = self.list_Ti_gap[alpha_g]
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g],current_Ti[j]), self.UN.T])
				tmp_left = tmp_left * Q_missing_byPatch
				tmp_left = tmp_left.T
				tmp_right = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[alpha_g], alpha_Ti[j]), self.UN.T ])
				# tmp_right = tmp_right * list_Q_gap[alpha_g]
				# tmp_right = tmp_right + current_A0[j]
				tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])						
				list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pgj_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pgj_patch])
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print(self.list_alpha)
		# end
		# compute beta

		return 0


	def interpolate_missing(self):
		# self.normalization()
		list_matrix = []
		for j in range(self.K):
			for g in range(len(self.markerwithgap)):
				tmp = self.list_beta[j] * self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_matrix.append(np.matmul(self.A1, tmp) * self.Q_gap[g])
		result = summatrix_list(list_matrix)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]
		# self.de_normalization()
		return final_result



class interpolation_gap_patch_2006():
	def __init__(self, reference, missing_matrix):
		missing_column = np.where(missing_matrix == 0)[1]
		missing_marker = np.unique(missing_column)
		list_transition = []
		list_predict = []
		for mm in missing_marker:
			mm2refer = np.copy(reference)
			mm2refer[np.where(missing_matrix[mm,:]==0)] = 0
			_, tmp_Usigma, tmp_U = np.linalg.svd(reference/np.sqrt(reference.shape[0]-1), full_matrices = False)
			UN = np.copy(tmp_U.T)
			_, tmp_U0sigma, tmp_U0 = np.linalg.svd(mm2refer/np.sqrt(mm2refer.shape[0]-1), full_matrices = False)
			UN0 = np.copy(tmp_U0.T)
			transitionMatrix = np.matmul(UN0.T , UN)
			list_transition.append(transitionMatrix)
			predict = matmul_list([mm2refer, UN0, transitionMatrix, UN.T])
			list_predict.append(predict)
		self.list_transition = list_transition
		self.list_predict = list_predict

class interpolation_weighted_gap_Yu_2006():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		# matrix F j g represents the transition matrix in case of missing gap g_th with reference is patch j
		self.missing_column = np.where(missing_matrix == 0)[1]
		self.missing_marker = np.unique(self.missing_column)
		self.reference_matrix = reference_matrix
		self.missing_matrix = missing_matrix
		self.leng = missing_matrix.shape[0]
		self.K = int(reference_matrix.shape[0] / missing_matrix.shape[0])
		self.F_matrix = []
		for patch_number in range(self.K):
			l = patch_number * self.leng
			r = l + self.leng
			current_patch = reference_matrix[l:r,:]
			self.F_matrix.append(interpolation_gap_patch_2006(current_patch, missing_matrix))
		self.compute()

	def compute(self):
		
		B_matrix = np.zeros(self.missing_matrix.shape)
		for patch_number in range(self.K):
			l = patch_number * self.leng
			r = l + self.leng
			current_patch = self.reference_matrix[l:r]
			mm_counter = 0
			for mm in self.missing_marker:
				B_matrix = B_matrix + current_patch - self.F_matrix[patch_number].list_predict[mm_counter]

		BTB = np.matmul(B_matrix.T, B_matrix)

		_, tmp_Usigma, tmp_U = np.linalg.svd(BTB/np.sqrt(BTB.shape[0]-1), full_matrices = False)
		k = setting_rank(tmp_Usigma)
		weightList = np.ones(tmp_Usigma.shape)
		print(k)
		for x in range(k):
			weightList[x] = 1.0/tmp_Usigma[x]
		self.W = np.diag(weightList)
		np.savetxt("check.txt", self.W, fmt = "%.2f")
		np.savetxt("check1.txt", tmp_Usigma, fmt = "%.2f")
		np.savetxt("check2.txt", BTB, fmt = "%.2f")
		stop
		p_AN = self.AN
		p_AN0 = self.AN0
		
		list_left_matrix_G = []
		self.list_Ti_gap = []
		self.list_UN0_gap = []
		
		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)
		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma)) - 1
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		counter_gap = 0
		
		# divide original A0 per patch
		list_A0_origin = []
		list_A_origin = []
		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A0_origin.append(np.copy(p_AN0[l:r]))
			list_A_origin.append(np.copy(p_AN[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		# end section
		Q_missing_byPatch = np.zeros(list_A0_origin[0].shape)
		for marker in self.markerwithgap:
			Q_missing_byPatch[:, marker*3: marker*3+3] = 1
		Q_deduct_missing = np.ones(list_A0_origin[0].shape)
		Q_deduct_missing[np.where(Q_missing_byPatch == 1)] = 0
		list_Q_gap = []	
		# prepare list_Q_gap 

		for marker in self.markerwithgap:
			tmp = np.zeros(list_A0_origin[0].shape)
			tmp[:, marker*3 : marker*3+3] = 1
			list_Q_gap.append(np.copy(tmp))
		# end
		self.Q_gap = []
		for marker in self.markerwithgap:
			tmp = np.zeros(self.A1.shape)
			tmp[:, marker*3 : marker*3+3] = 1
			self.Q_gap.append(tmp)

		# divide A0 per patch that regard to missing gaps

		list_A0_gaps = []
		# list_A0_gaps contains all the A0 that miss gap respectively
		for marker in self.markerwithgap:
			list_A0 = []
			#  list_A0 contains the A0 missing only gap marker
			r = len(self.AN0)
			l = r - self.fix_leng
			
			AN0_gapG = self.list_N0_gap[counter_gap]
			_, tmp_U0sigma, tmp_U0 = np.linalg.svd(AN0_gapG/np.sqrt(AN0_gapG.shape[0]-1), full_matrices = False)
			current_UN0 = np.copy(tmp_U0.T)[:, :ksmall]
			self.list_UN0_gap.append(current_UN0)
			# compute UN0 corresponds to the current missing marker g

			while l >= 0:
				list_A0.append(np.copy(AN0_gapG[l:r]))
				l -= self.fix_leng
				r -= self.fix_leng

			list_A0_gaps.append(list_A0)
		
			list_Ti = []

			for patch_number in range(self.K):
				AiUN = np.matmul(list_A_origin[patch_number], self.UN)
				Ai0UN0 = np.matmul(list_A0[patch_number], current_UN0)
				
				X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
				list_Ti.append(np.copy(X[0]))

				# list_Ti.append(np.matmul(self.UN.T, current_UN0))
			
			# compute Ti corresponds to current missing marker g
				
			self.list_Ti_gap.append(list_Ti)		
	
			list_P = []
			for patch_number in range(self.K):
				tmp = list_A_origin[patch_number] - np.matmul(list_A0[patch_number], 
						np.matmul(np.matmul(current_UN0, list_Ti[patch_number]), self.UN.T))
				list_P.append(tmp)
			P_matrix = summatrix_list(list_P)
			P_matrix = P_matrix * list_Q_gap[counter_gap]

			joint_number = P_matrix.shape[1]
			list_P_column = []
			for i in range(joint_number):
				left_P = P_matrix[:,i]
				for j in range(joint_number):
					right_P = P_matrix[:,j]
					list_P_column.append(np.copy(left_P * right_P))
			left_matrix_tmp = np.vstack(list_P_column)
			list_left_matrix_G.append(left_matrix_tmp)
			counter_gap	+= 1 # increase index in list_N0_gap and list_Q_gap
		# end section
		# compute weight	

		left_matrix = summatrix_list(list_left_matrix_G)
		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# end
		# compute alpha
		
		Q_missing_byPatch = np.ones(Q_missing_byPatch.shape)

		stop
		list_Qgj = []
		for g in range(len(self.markerwithgap)):
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g], current_Ti[j]), self.UN.T])
				tmp_left = tmp_left * list_Q_gap[g]
				tmp_left = tmp_left + current_A0[j]
				tmpQ =  matmul_list([tmp_left.T, self.W, list_A_origin[j]])
				list_Qgj.append(tmpQ)
		right_form = summatrix_list(list_Qgj)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		stop
		list_Pgj_patch = []
		for alpha_g in range(len(self.markerwithgap)):
			list_tmp = []
			alpha_Ti = self.list_Ti_gap[alpha_g]
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g],current_Ti[j]), self.UN.T])
				tmp_left = tmp_left * Q_missing_byPatch
				tmp_left = tmp_left.T
				tmp_right = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[alpha_g], alpha_Ti[j]), self.UN.T ])
				# tmp_right = tmp_right * list_Q_gap[alpha_g]
				# tmp_right = tmp_right + current_A0[j]
				tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])						
				list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pgj_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pgj_patch])
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print(self.list_alpha)
		# end
		# compute beta

		return 0


	def interpolate_missing(self):
		# self.normalization()
		list_matrix = []
		for j in range(self.K):
			for g in range(len(self.markerwithgap)):
				tmp = self.list_beta[j] * self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_matrix.append(np.matmul(self.A1, tmp) * self.Q_gap[g])
		result = summatrix_list(list_matrix)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]
		# self.de_normalization()
		return final_result




class interpolation_weighted_gap_dang():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries, self.list_N0_gap = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		self.compute_svd()
		self.result_norm = self.interpolate_missing()
		self.de_normalization()

	def normalization(self):
		AA = np.copy(self.combine_matrix)
		weightScale = 200
		MMweight = 0.02
		[frames, columns] = AA.shape
		columnindex = np.where(AA == 0)[1]
		frameindex = np.where(AA == 0)[0]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		self.markerwithgap = markerwithgap
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
		downsample = True
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
		weight_vector = np.min(weight_vector, 0)
		weight_vector = np.exp(np.divide(-np.square(weight_vector),(2*np.square(weightScale))))
		weight_vector[markerwithgap] = MMweight
		M_zero = np.copy(Data)
		# N_nogap = np.copy(Data[:Data.shape[0]-AA1.shape[0]])
		N_nogap = np.delete(Data, framewithgap, 0)
		N_nogap_origin = np.copy(N_nogap)
		N_zero = np.copy(N_nogap)
		N_zero[:,columnwithgap] = 0

		mean_N_nogap = np.mean(N_nogap, 0)
		mean_N_nogap = mean_N_nogap.reshape((1, mean_N_nogap.shape[0]))

		mean_N_zero = np.mean(N_zero, 0)
		tmp_tmp = np.copy(mean_N_zero)
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
		list_N0_gap = []
		# looping for gaps
		for marker in markerwithgap:
			N_zero_tmp = np.copy(N_nogap_origin)
			N_zero_tmp[:,marker*3 : marker*3+3] = 0

			mean_N_zero_tmp = np.mean(N_zero_tmp, 0)
			mean_N_zero_tmp = mean_N_zero_tmp.reshape((1, mean_N_zero_tmp.shape[0]))
			m6 = np.ones((N_zero_tmp.shape[0],1))*mean_N_zero_tmp

			N_zero_tmp = np.multiply(((N_zero_tmp-m6) / m5),m33)
			list_N0_gap.append(N_zero_tmp)
		return [M_zero, N_nogap, N_zero], [m7, stdev_N_no_gaps, column_weight, MeanMat], list_N0_gap


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
		
		list_left_matrix_G = []
		self.list_Ti_gap = []
		self.list_UN0_gap = []
		
		_, tmp_Usigma, tmp_U = np.linalg.svd(p_AN/np.sqrt(p_AN.shape[0]-1), full_matrices = False)
		self.UN = np.copy(tmp_U.T)
		_, tmp_U0sigma, tmp_U0 = np.linalg.svd(p_AN0/np.sqrt(p_AN0.shape[0]-1), full_matrices = False)
		self.UN0 = np.copy(tmp_U0.T)

		ksmall = max(setting_rank(tmp_Usigma), setting_rank(tmp_U0sigma)) - 1
		
		self.UN = self.UN[:, :ksmall]
		self.UN0 = self.UN0[:, :ksmall]
		counter_gap = 0
		
		# divide original A0 per patch
		list_A0_origin = []
		list_A_origin = []
		r = len(self.AN0)
		l = r - self.fix_leng

		while l >= 0:
			list_A0_origin.append(np.copy(p_AN0[l:r]))
			list_A_origin.append(np.copy(p_AN[l:r]))
			l -= self.fix_leng
			r -= self.fix_leng

		# end section
		Q_missing_byPatch = np.zeros(list_A0_origin[0].shape)
		for marker in self.markerwithgap:
			Q_missing_byPatch[:, marker*3: marker*3+3] = 1
		Q_deduct_missing = np.ones(list_A0_origin[0].shape)
		Q_deduct_missing[np.where(Q_missing_byPatch == 1)] = 0
		list_Q_gap = []	
		# prepare list_Q_gap 

		for marker in self.markerwithgap:
			tmp = np.zeros(list_A0_origin[0].shape)
			tmp[:, marker*3 : marker*3+3] = 1
			list_Q_gap.append(np.copy(tmp))
		# end
		self.Q_gap = []
		for marker in self.markerwithgap:
			tmp = np.zeros(self.A1.shape)
			tmp[:, marker*3 : marker*3+3] = 1
			self.Q_gap.append(tmp)

		# divide A0 per patch that regard to missing gaps

		list_A0_gaps = []
		# list_A0_gaps contains all the A0 that miss gap respectively
		for marker in self.markerwithgap:
			list_A0 = []
			#  list_A0 contains the A0 missing only gap marker
			r = len(self.AN0)
			l = r - self.fix_leng
			
			AN0_gapG = self.list_N0_gap[counter_gap]
			_, tmp_U0sigma, tmp_U0 = np.linalg.svd(AN0_gapG/np.sqrt(AN0_gapG.shape[0]-1), full_matrices = False)
			current_UN0 = np.copy(tmp_U0.T)[:, :ksmall]
			self.list_UN0_gap.append(current_UN0)
			# compute UN0 corresponds to the current missing marker g

			while l >= 0:
				list_A0.append(np.copy(AN0_gapG[l:r]))
				l -= self.fix_leng
				r -= self.fix_leng

			list_A0_gaps.append(list_A0)
		
			list_Ti = []

			for patch_number in range(self.K):
				AiUN = np.matmul(list_A_origin[patch_number], self.UN)
				Ai0UN0 = np.matmul(list_A0[patch_number], current_UN0)
				
				X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
				list_Ti.append(np.copy(X[0]))

				# list_Ti.append(np.matmul(self.UN.T, current_UN0))
			
			# compute Ti corresponds to current missing marker g
				
			self.list_Ti_gap.append(list_Ti)		
	
			list_P = []
			for patch_number in range(self.K):
				tmp = list_A_origin[patch_number] - np.matmul(list_A0[patch_number], 
						np.matmul(np.matmul(current_UN0, list_Ti[patch_number]), self.UN.T))
				list_P.append(tmp)
			P_matrix = summatrix_list(list_P)
			P_matrix = P_matrix * list_Q_gap[counter_gap]

			joint_number = P_matrix.shape[1]
			list_P_column = []
			for i in range(joint_number):
				left_P = P_matrix[:,i]
				for j in range(joint_number):
					right_P = P_matrix[:,j]
					list_P_column.append(np.copy(left_P * right_P))
			left_matrix_tmp = np.vstack(list_P_column)
			list_left_matrix_G.append(left_matrix_tmp)
			counter_gap	+= 1 # increase index in list_N0_gap and list_Q_gap
		# end section
		# compute weight	
		left_matrix = summatrix_list(list_left_matrix_G)
		u, d, v = np.linalg.svd(np.matmul(left_matrix.T, left_matrix))
		v = v.T
		weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# end
		# compute alpha
		

		list_Qgj = []
		for g in range(len(self.markerwithgap)):
			current_A0 = list_A0_origin
			current_Ti = self.list_Ti_gap[g]
			for j in range(self.K):
				tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g], current_Ti[j]), self.UN.T])
				# tmp_left = tmp_left * list_Q_gap[g]
				# tmp_left = tmp_left + current_A0[j]
				tmpQ =  matmul_list([tmp_left.T, self.W, list_A_origin[j]])
				list_Qgj.append(tmpQ)
		right_form = summatrix_list(list_Qgj)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		
		list_Pgj_patch = []
		for alpha_g in range(len(self.markerwithgap)):
			list_tmp = []
			alpha_Ti = self.list_Ti_gap[alpha_g]
			for g in range(len(self.markerwithgap)):
				current_A0 = list_A0_origin
				current_Ti = self.list_Ti_gap[g]
				for j in range(self.K):
					tmp_left = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[g],current_Ti[j]), self.UN.T])
					# tmp_left = tmp_left * list_Q_gap[g]
					# tmp_left = tmp_left + current_A0[j]
					tmp_right = matmul_list([current_A0[j], np.matmul(self.list_UN0_gap[alpha_g], alpha_Ti[j]), self.UN.T ])
					# tmp_right = tmp_right * list_Q_gap[alpha_g]
					# tmp_right = tmp_right + current_A0[j]
					tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])						
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pgj_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pgj_patch])
		# self.list_alpha = np.linalg.lstsq(left_form, right_form, rcond = None)[0]
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print("list alpha")
		print(self.list_alpha)
		# end
		# compute beta

		list_Zik = []
		for beta_loop in range(self.K):
			for i in range(self.K):
				# compute P betaloop inside
				list_tmp = []
				for g_inP in range(len(self.markerwithgap)):
					tmp_ploop = self.list_alpha[g_inP] * np.matmul(
						np.matmul(self.list_UN0_gap[g_inP], self.list_Ti_gap[g_inP][beta_loop]), self.UN.T)
					tmp_ploop = np.matmul(list_A0_origin[i], tmp_ploop) 
					# tmp_ploop = tmp_ploop * list_Q_gap[g_inP]
					list_tmp.append(tmp_ploop)

				tmp_left = summatrix_list(list_tmp)
				# tmp_left = tmp_left + list_A0_origin[i]
				# using precomputed P betaloop
				tmp = matmul_list([tmp_left.T, self.W, list_A_origin[i]])
				list_Zik.append(np.copy(tmp))
		right_form = summatrix_list(list_Zik)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		list_Yik_beta = []
		# betaindex : beta1, beta2, .. k beta
		# betaloop : beta1 1st time, 2nd time ... k times
		for beta_index in range(self.K):
			list_tmp = []
			for beta_loop in range(self.K):
				for i in range(self.K):
					# compute P betaloop
					list_tmp_left = []
					for g_inP in range(len(self.markerwithgap)):
						tmp_matrix = self.list_alpha[g_inP] * np.matmul(
							np.matmul(self.list_UN0_gap[g_inP], self.list_Ti_gap[g_inP][beta_loop]), self.UN.T)
						tmp_matrix = np.matmul(list_A0_origin[i], tmp_matrix) 
						list_tmp_left.append(tmp_matrix)

					tmp_left = summatrix_list(list_tmp_left) 
					tmp_left = tmp_left.T
					# compute P beta index
					list_tmp_right = []
					for g_inP in range(len(self.markerwithgap)):
						tmp_matrix = self.list_alpha[g_inP] * np.matmul(
							np.matmul(self.list_UN0_gap[g_inP], self.list_Ti_gap[g_inP][beta_index]), self.UN.T)
						tmp_matrix = np.matmul(list_A0_origin[i], tmp_matrix)
						list_tmp_right.append(tmp_matrix)
					tmp_right = summatrix_list(list_tmp_right) 

					tmp_P = matmul_list( [tmp_left, self.W, tmp_right])	
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Yik_beta.append(tmp.reshape(xx*yy, 1))
		left_form = np.hstack([ x for x in list_Yik_beta])

		self.list_beta = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print("list beta")
		print(self.list_beta)
		return 0


	def interpolate_missing(self):
		# self.normalization()
		# self.list_alpha = [0.95778174]
		# self.list_beta = [-659.95899244, 1046.9425573, -344.58085942, -41.40273387]
		list_matrix = []
		for j in range(self.K):
			for g in range(len(self.markerwithgap)):
				tmp = self.list_beta[j] * self.list_alpha[g] * np.matmul(np.matmul(self.list_UN0_gap[g], self.list_Ti_gap[g][j]), self.UN.T)
				list_matrix.append(np.matmul(self.A1, tmp))
		result = summatrix_list(list_matrix)
		final_result = np.copy(self.A1)
		final_result[np.where(self.A1 == 0)] = result[np.where(self.A1 == 0)]
		# self.de_normalization()
		return final_result

class interpolation_weighted_gap_dang_v2():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		self.missing_matrix = missing_matrix
		self.reference_matrix = reference_matrix
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.original_combine = np.copy(self.combine_matrix)
		self.fix_leng = missing_matrix.shape[0]
		self.original_refer = reference_matrix
		self.original_missing = missing_matrix
		self.F_matrix = self.prepare()
		self.missing_matrix = self.norm_Data[-self.missing_matrix.shape[0]:]
		self.reference_matrix = self.norm_Data[:-self.missing_matrix.shape[0]]
		# F_matrix compute multiplication of A0j U0N Tj U(N) regards to missing only gap g
		# F_matrix contains matries having same size with A(j) original
		# Fgij = value of missing gap g and using patch i to interpolate patch j
		self.compute_variant()
		self.result_norm = self.interpolate_missing()

	def prepare(self):
		list_F_matrix = []
		DistalThreshold = 0.5
		test_data = np.copy(self.missing_matrix)
		source_data = np.copy(self.reference_matrix)

		AA = np.copy(self.combine_matrix)
		columnindex = np.where(AA == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		# print(markerwithgap)
		self.markerwithgap = markerwithgap
		missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
		list_frame = np.arange(test_data.shape[0])
		full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
		self.full_frame_testdata = full_frame_testdata
		
		[frames, columns] = AA.shape
		Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		# np.savetxt("checkMeanmat.txt", self.MeanMat[-self.missing_matrix.shape[0]:], fmt = "%.2f")
		AA = AA - self.MeanMat
		AA[np.where(self.combine_matrix == 0)] = 0
		self.norm_Data = np.copy(AA)
		#  /////////////////////////////////////////////////////////////////////////
		# note recontruct Meanmat
		self.K = self.reference_matrix.shape[0] + len(full_frame_testdata)
		self.K = int(self.K / self.fix_leng)
		self.list_mark_matrixQ = []
		for marker in markerwithgap:
			tmp = np.zeros(self.missing_matrix.shape)
			missing_frame_marker = np.where(self.missing_matrix[:, marker*3] == 0)
			tmp[missing_frame_marker, marker*3 : marker*3+3] = 1
			self.list_mark_matrixQ.append(np.copy(tmp))

		self.Q_matrix = np.zeros(self.missing_matrix.shape)
		self.Q_matrix[np.where(self.missing_matrix == 0)] = 1

		list_frameidx_patch = get_list_frameidx_patch(self.K, self.fix_leng, len(full_frame_testdata))
		self.list_frameidx_patch = list_frameidx_patch
		counter_marker = 0
		for marker in markerwithgap:
			# print(marker)
			missing_frame = np.where(test_data[:, marker*3] == 0)
			# print(missing_frame)
			EuclDist2Marker = compute_weight_vect_norm([marker], AA)
			# print(EuclDist2Marker)
			thresh = np.mean(EuclDist2Marker) * DistalThreshold
			# print(thresh)
			# stop
			Data_remove_joint = np.copy(AA)
			for sub_marker in range(len(EuclDist2Marker)):
				if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
					Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			Data_remove_joint[:, marker*3:marker*3+3] = np.copy(AA[:, marker*3:marker*3+3])
			frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
			for sub_marker in markerwithgap:
				if sub_marker != marker:
					if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
						# print(marker, " affect ", sub_marker)
						Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			
			source_data_renew = Data_remove_joint[: len(self.reference_matrix)]
			full_frame_renew = full_frame_testdata + self.reference_matrix.shape[0]
			full_joint_testingData = Data_remove_joint[full_frame_renew, :]
			only_missing_testingData = Data_remove_joint[frames_missing_marker, :]
			Data_renew =  np.vstack([source_data_renew, full_joint_testingData])
			full_testData = np.vstack([Data_renew, only_missing_testingData])
			# np.savetxt("checkinput.txt", Data_remove_joint, fmt = "%.2f")
			# stop
			gap_interpolation = interpolation_gap_patch_v2(Data_renew, marker, self.missing_matrix, list_frameidx_patch, full_testData, Data_remove_joint, np.copy(AA))
			list_F_matrix.append(gap_interpolation)
			counter_marker += 1

		return list_F_matrix

	def compute_variant(self):
		full_joint_testingData = self.missing_matrix[self.full_frame_testdata, :]
		Data_renew =  np.vstack([self.reference_matrix, full_joint_testingData])
		list_left_matrix_G = []
		counter_gap = 0
		for marker in self.markerwithgap:
			list_P = []
			for patch_number in range(self.K):
				result_precJJ = self.F_matrix[counter_gap].Predic[patch_number][patch_number]
				A_patch = Data_renew[self.list_frameidx_patch[patch_number][0]: self.list_frameidx_patch[patch_number][1]]
				tmp = (A_patch  * self.list_mark_matrixQ[counter_gap]) - result_precJJ
				list_P.append(tmp)
			P_matrix = summatrix_list(list_P) 

			joint_number = P_matrix.shape[1]
			list_P_column = []
			for i in range(joint_number):
				left_P = P_matrix[:,i]
				for j in range(joint_number):
					right_P = P_matrix[:,j]
					list_P_column.append(np.copy(left_P * right_P))
			left_matrix_tmp = np.vstack(list_P_column)
			list_left_matrix_G.append(left_matrix_tmp)
			counter_gap	+= 1 # increase index in list_N0_gap and list_Q_gap
		# end section
		# compute weight	

		left_matrix = summatrix_list(list_left_matrix_G)
		u, d, v = np.linalg.svd(np.matmul(left_matrix.T, left_matrix))
		# u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		if weight_list[-1] == 1:
			# u, d, v = np.linalg.svd(np.matmul(left_matrix.T, left_matrix))
			u, d, v = np.linalg.svd(left_matrix)
			v = v.T
			weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# np.savetxt("checkw_vector.txt", self.W, fmt = "%.2f")
		# end
		# compute alpha

		list_Qgj = []
		for g in range(len(self.markerwithgap)):
			for j in range(self.K):
				tmp_left = self.F_matrix[g].Predic[j][j]
				Aj =  Data_renew[self.list_frameidx_patch[j][0]: self.list_frameidx_patch[j][1]]
				tmpQ =  matmul_list([tmp_left.T, self.W, (Aj * self.Q_matrix)])
				list_Qgj.append(tmpQ)
		right_form = summatrix_list(list_Qgj)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)
		
		list_Pgj_patch = []
		for alpha_g in range(len(self.markerwithgap)):
			list_tmp = []
			for g in range(len(self.markerwithgap)):
				for j in range(self.K):
					tmp_left = self.F_matrix[g].Predic[j][j]
					tmp_right = self.F_matrix[alpha_g].Predic[j][j]
					tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])						
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pgj_patch.append(tmp.reshape(xx*yy, 1))

		left_form = np.hstack([ x for x in list_Pgj_patch])
		# np.savetxt("checkleft.txt", left_form, fmt = "%.8f")
		# np.savetxt("checkright.txt", right_form, fmt = "%.8f")
		# self.list_alpha = np.linalg.lstsq(left_form, right_form, rcond = None)[0]
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print("list alpha")
		# if 
		self.list_alpha = [1] * len(self.markerwithgap)
		print(self.list_alpha)

		# end
		# compute beta
		
		list_Zik = []
		for beta_loop in range(self.K):
			for i in range(self.K):
				# compute P betaloop inside
				list_tmp = []
				for g_inP in range(len(self.markerwithgap)):
					tmp_ploop = self.list_alpha[g_inP] * self.F_matrix[g_inP].Predic[beta_loop][i]
					list_tmp.append(tmp_ploop)

				tmp_left = summatrix_list(list_tmp)
				Ai = Data_renew[self.list_frameidx_patch[i][0]: self.list_frameidx_patch[i][1]]
				tmp = matmul_list([tmp_left.T, self.W, (Ai * self.Q_matrix)])
				list_Zik.append(np.copy(tmp))
		
		right_form = summatrix_list(list_Zik)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)

		list_Yik_beta = []
		# betaindex : beta1, beta2, .. k beta
		# betaloop : beta1 1st time, 2nd time ... k times
		for beta_index in range(self.K):
			list_tmp = []
			for beta_loop in range(self.K):
				for i in range(self.K):
					# compute P betaloop
					list_tmp_left = []
					for g_inP in range(len(self.markerwithgap)):
						tmp_matrix = self.list_alpha[g_inP] * self.F_matrix[g_inP].Predic[beta_loop][i]
						list_tmp_left.append(tmp_matrix)

					tmp_left = summatrix_list(list_tmp_left) 
					# compute P beta index
					list_tmp_right = []
					for g_inP in range(len(self.markerwithgap)):
						tmp_matrix = self.list_alpha[g_inP] * self.F_matrix[g_inP].Predic[beta_index][i]
						list_tmp_right.append(tmp_matrix)
					tmp_right = summatrix_list(list_tmp_right) 

					tmp_P = matmul_list( [tmp_left.T, self.W, tmp_right])	
					list_tmp.append(tmp_P)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Yik_beta.append(tmp.reshape(xx*yy, 1))
		left_form = np.hstack([ x for x in list_Yik_beta])

		self.list_beta = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		print("list beta")
		print(self.list_beta)
		print(np.sum(self.list_beta))
		if np.sum(self.list_beta) <= 0.00001:
			self.list_beta = [1.0/ self.K] * self.K
		return 0


	def interpolate_missing(self):
		# self.list_beta = [ 0.03862384, 0.03669788, -0.03202568, 0.95670474]
		# self.list_alpha[0] = 1
		# print("self.list_beta sum: ", np.sum(self.list_beta))
		# counter = 0
		list_matrix = []
		for j in range(1):
			for g in range(len(self.markerwithgap)):
				list_matrix.append(self.F_matrix[g].interpolate_A[0])
				# Ajg = self.F_matrix[g].interpolate_A[j]
				# tmp =  self.list_beta[j] * self.list_alpha[g] * np.copy(Ajg)
				# list_matrix.append(tmp)
		# 		# counter += self.list_alpha[g] * self.list_beta[j]
		# 		# np.savetxt("checkresult"+ str(g) +".txt", Ajg, fmt = "%.2f")
		result = summatrix_list(list_matrix)
		result = result[-self.missing_matrix.shape[0]:] + self.MeanMat[-self.missing_matrix.shape[0]:]
		# np.savetxt("checkresult.txt", result, fmt = "%.2f")
		# stop
		# print("counter: ", counter)
		# result = self.F_matrix[0].reconstruct + self.MeanMat
		# result = result[-self.missing_matrix.shape[0]:]

		final_result = np.copy(self.original_missing)
		final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
		return final_result



class interpolation_weighted_gap_dang_v3():
	def __init__(self, reference_matrix, missing_matrix):
		# this function is integrated with norm so no norm option as previous approaches
		self.missing_matrix = missing_matrix
		self.reference_matrix = reference_matrix
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.original_combine = np.copy(self.combine_matrix)
		self.fix_leng = missing_matrix.shape[0]
		self.original_refer = reference_matrix
		self.original_missing = missing_matrix
		self.F_matrix = self.prepare()
		self.missing_matrix = self.norm_Data[-self.missing_matrix.shape[0]:]
		self.reference_matrix = self.norm_Data[:-self.missing_matrix.shape[0]]
		# F_matrix compute multiplication of A0j U0N Tj U(N) regards to missing only gap g
		# F_matrix contains matries having same size with A(j) original
		# Fgij = value of missing gap g and using patch i to interpolate patch j
		self.compute_variant()
		self.result_norm = self.interpolate_missing()

	def prepare(self):
		list_F_matrix = []
		DistalThreshold = 0.5
		test_data = np.copy(self.missing_matrix)
		source_data = np.copy(self.reference_matrix)

		AA = np.copy(self.combine_matrix)
		columnindex = np.where(AA == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		# print(markerwithgap)
		self.markerwithgap = markerwithgap
		missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
		list_frame = np.arange(test_data.shape[0])
		full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
		self.full_frame_testdata = full_frame_testdata
		
		[frames, columns] = AA.shape
		Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		# np.savetxt("checkMeanmat.txt", self.MeanMat[-self.missing_matrix.shape[0]:], fmt = "%.2f")
		AA = AA - self.MeanMat
		AA[np.where(self.combine_matrix == 0)] = 0
		self.norm_Data = np.copy(AA)
		#  /////////////////////////////////////////////////////////////////////////
		# note recontruct Meanmat
		self.K = self.reference_matrix.shape[0] + len(full_frame_testdata)
		self.K = int(self.K / self.fix_leng)
		self.list_mark_matrixQ = []
		self.ksmall = 0
		for marker in markerwithgap:
			tmp = np.zeros(self.missing_matrix.shape)
			missing_frame_marker = np.where(self.missing_matrix[:, marker*3] == 0)
			tmp[missing_frame_marker, marker*3 : marker*3+3] = 1
			self.list_mark_matrixQ.append(np.copy(tmp))

		self.Q_matrix = np.zeros(self.missing_matrix.shape)
		self.Q_matrix[np.where(self.missing_matrix == 0)] = 1

		list_frameidx_patch = get_list_frameidx_patch(self.K, self.fix_leng, len(full_frame_testdata))
		self.list_frameidx_patch = list_frameidx_patch
		counter_marker = 0
		for marker in markerwithgap:
			# print(marker)
			missing_frame = np.where(test_data[:, marker*3] == 0)
			# print(missing_frame)
			EuclDist2Marker = compute_weight_vect_norm([marker], AA)
			# print(EuclDist2Marker)
			thresh = np.mean(EuclDist2Marker) * DistalThreshold
			# print(thresh)
			# stop
			Data_remove_joint = np.copy(AA)
			for sub_marker in range(len(EuclDist2Marker)):
				if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
					Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			Data_remove_joint[:, marker*3:marker*3+3] = np.copy(AA[:, marker*3:marker*3+3])
			frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
			for sub_marker in markerwithgap:
				if sub_marker != marker:
					if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
						# print(marker, " affect ", sub_marker)
						Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			
			source_data_renew = Data_remove_joint[: len(self.reference_matrix)]
			full_frame_renew = full_frame_testdata + self.reference_matrix.shape[0]
			full_joint_testingData = Data_remove_joint[full_frame_renew, :]
			only_missing_testingData = Data_remove_joint[frames_missing_marker, :]
			Data_renew =  np.vstack([source_data_renew, full_joint_testingData])
			full_testData = np.vstack([Data_renew, only_missing_testingData])
			# np.savetxt("checkinput.txt", Data_remove_joint, fmt = "%.2f")
			# stop
			gap_interpolation = interpolation_gap_patch_v2(Data_renew, marker, self.missing_matrix, list_frameidx_patch, full_testData, Data_remove_joint, np.copy(AA))
			self.ksmall = max(self.ksmall, gap_interpolation.ksmall)
			list_F_matrix.append(gap_interpolation)
			counter_marker += 1
		return list_F_matrix

	def compute_variant(self):
		full_joint_testingData = self.missing_matrix[self.full_frame_testdata, :]
		Data_renew =  np.vstack([self.reference_matrix, full_joint_testingData])
		# compute weight
	
		list_left_matrix = []
		for patch_number in range(self.K):
			list_matrix = []
			for g in range(len(self.markerwithgap)):
				list_matrix.append(self.F_matrix[g].Predic[patch_number][patch_number])
			prec_matrix = summatrix_list(list_matrix)
			Aj =  Data_renew[self.list_frameidx_patch[patch_number][0]: self.list_frameidx_patch[patch_number][1]]
			current_patch = np.copy(Aj * self.Q_matrix) - prec_matrix
			for column in range(self.ksmall):
				for clm in range(self.ksmall):
					tmp = np.multiply(current_patch[:, column], current_patch[:, clm])
					list_left_matrix.append(tmp)


		left_matrix = np.vstack(list_left_matrix)
		u, d, v = np.linalg.svd(left_matrix)
		v = v.T
		weight_list = v[:, -1]
		if weight_list[-1] == 1:
			u, d, v = np.linalg.svd(np.matmul(left_matrix.T, left_matrix))
			# u, d, v = np.linalg.svd(left_matrix)
			v = v.T
			weight_list = v[:, -1]
		self.W = np.diag(weight_list)
		# compute alpha
		list_Qjk = []
		for j in range(self.K):
			for h in range(self.K) :
				list_matrix = []
				for g in range(len(self.markerwithgap)):
					list_matrix.append(self.F_matrix[g].Predic[h][j])
				A0U0TUN = summatrix_list(list_matrix)
				Aj =  Data_renew[self.list_frameidx_patch[j][0]: self.list_frameidx_patch[j][1]]
				tmpQ = matmul_list([A0U0TUN.T, self.W, (Aj * self.Q_matrix)])
				list_Qjk.append(tmpQ)
		right_form = summatrix_list(list_Qjk)
		xx, yy = right_form.shape
		right_form = right_form.reshape(xx*yy, 1)
		

		list_Pij_patch = []
		for patch_number in range(self.K):
			list_tmp = []
			for j in range(self.K):
				for h in range(self.K):
					list_matrix = []
					for g in range(len(self.markerwithgap)):
						list_matrix.append(self.F_matrix[g].Predic[h][j])
					Ai_left = summatrix_list(list_matrix)

					list_matrix = []
					for g in range(len(self.markerwithgap)):
						list_matrix.append(self.F_matrix[g].Predic[patch_number][j])
					Ai_right = summatrix_list(list_matrix)

					tmpP = matmul_list([Ai_left.T, self.W, Ai_right])
					list_tmp.append(tmpP)
			tmp = summatrix_list(list_tmp)
			xx, yy = tmp.shape
			list_Pij_patch.append(tmp.reshape(xx*yy, 1))
		left_form = np.hstack([ x for x in list_Pij_patch])
		list_zero = []
		for i in range(left_form.shape[0]):
			check = True
			for j in range(left_form.shape[1]):
				if left_form[i][j] >= 0.00001: 
					check = False
					break
			if check: list_zero.append(i)
		print(len(list_zero))
		arr = np.arange(self.K*3)
		np.random.shuffle(arr)
		left_form = np.delete(left_form, list_zero, 0)
		right_form = np.delete(right_form, list_zero, 0)
		if left_form.shape[0] >= self.K * 3:
			left_form = left_form[arr]
			right_form = right_form[arr]
		# self.list_alpha = np.linalg.lstsq(left_form, right_form, rcond = None)[0]
		self.list_alpha = np.linalg.lstsq(np.matmul(left_form.T, left_form), np.matmul(left_form.T, right_form), rcond = None)[0]
		if np.sum(self.list_alpha) <= 0.00001:
			self.list_alpha = [1.0 / self.K] * self.K
		# self.list_alpha[0] += self.list_alpha[1]
		# self.list_alpha[1] = 0
		print(self.list_alpha)
		return self.list_alpha


	def interpolate_missing(self):
		list_matrix = []
		for j in range(self.K):
			predictAj = []
			for g in range(len(self.markerwithgap)):
				# list_matrix.append(self.F_matrix[g].interpolate_A[0])
				predictAj.append(self.F_matrix[g].interpolate_A[j])
			tmp = self.list_alpha[j] * summatrix_list(predictAj)
			list_matrix.append(tmp)
		result = summatrix_list(list_matrix)
		result = result[-self.missing_matrix.shape[0]:] + self.MeanMat[-self.missing_matrix.shape[0]:]

		final_result = np.copy(self.original_missing)
		final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
		return final_result



class interpolation_weighted_gap_dang_v4():
	def __init__(self, reference_matrix, missing_matrix):
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.normed_matries, self.reconstruct_matries = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		self.compute_svd()
		self.missing_matrix = missing_matrix
		self.reference_matrix = reference_matrix
		self.original_combine = np.copy(self.combine_matrix)
		self.original_refer = reference_matrix
		self.original_missing = missing_matrix
		self.F_matrix = self.prepare()
		self.missing_matrix = self.norm_Data[-self.missing_matrix.shape[0]:]
		self.reference_matrix = self.norm_Data[:-self.missing_matrix.shape[0]]
		# F_matrix compute multiplication of A0j U0N Tj U(N) regards to missing only gap g
		# F_matrix contains matries having same size with A(j) original
		# Fgij = value of missing gap g and using patch i to interpolate patch j
		self.result_norm = self.interpolate_missing()

	def prepare(self):
		list_F_matrix = []
		DistalThreshold = 0.5
		test_data = np.copy(self.missing_matrix)
		source_data = np.copy(self.reference_matrix)

		AA = np.copy(self.combine_matrix)
		columnindex = np.where(AA == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		# print(markerwithgap)
		self.markerwithgap = markerwithgap
		missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
		list_frame = np.arange(test_data.shape[0])
		full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
		self.full_frame_testdata = full_frame_testdata
		
		[frames, columns] = AA.shape
		Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		# np.savetxt("checkMeanmat.txt", self.MeanMat[-self.missing_matrix.shape[0]:], fmt = "%.2f")
		AA = AA - self.MeanMat
		AA[np.where(self.combine_matrix == 0)] = 0
		self.norm_Data = np.copy(AA)
		#  /////////////////////////////////////////////////////////////////////////
		# note recontruct Meanmat
		self.K = self.reference_matrix.shape[0] + len(full_frame_testdata)
		self.K = int(self.K / self.fix_leng)
		self.list_mark_matrixQ = []
		for marker in markerwithgap:
			tmp = np.zeros(self.missing_matrix.shape)
			missing_frame_marker = np.where(self.missing_matrix[:, marker*3] == 0)
			tmp[missing_frame_marker, marker*3 : marker*3+3] = 1
			self.list_mark_matrixQ.append(np.copy(tmp))

		self.Q_matrix = np.zeros(self.missing_matrix.shape)
		self.Q_matrix[np.where(self.missing_matrix == 0)] = 1

		list_frameidx_patch = get_list_frameidx_patch(self.K, self.fix_leng, len(full_frame_testdata))
		self.list_frameidx_patch = list_frameidx_patch
		counter_marker = 0
		for marker in markerwithgap:
			# print(marker)
			missing_frame = np.where(test_data[:, marker*3] == 0)
			# print(missing_frame)
			EuclDist2Marker = compute_weight_vect_norm([marker], AA)
			# print(EuclDist2Marker)
			thresh = np.mean(EuclDist2Marker) * DistalThreshold
			# print(thresh)
			# stop
			Data_remove_joint = np.copy(AA)
			for sub_marker in range(len(EuclDist2Marker)):
				if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
					Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			Data_remove_joint[:, marker*3:marker*3+3] = np.copy(AA[:, marker*3:marker*3+3])
			frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
			for sub_marker in markerwithgap:
				if sub_marker != marker:
					if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
						# print(marker, " affect ", sub_marker)
						Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			
			source_data_renew = Data_remove_joint[: len(self.reference_matrix)]
			full_frame_renew = full_frame_testdata + self.reference_matrix.shape[0]
			full_joint_testingData = Data_remove_joint[full_frame_renew, :]
			only_missing_testingData = Data_remove_joint[frames_missing_marker, :]
			Data_renew =  np.vstack([source_data_renew, full_joint_testingData])
			full_testData = np.vstack([Data_renew, only_missing_testingData])
			# np.savetxt("checkinput.txt", Data_remove_joint, fmt = "%.2f")
			# stop
			gap_interpolation = interpolation_gap_patch_v2(Data_renew, marker, self.missing_matrix, list_frameidx_patch, full_testData, Data_remove_joint, np.copy(AA))
			list_F_matrix.append(gap_interpolation)
			counter_marker += 1
		return list_F_matrix

	def normalization(self):
		normed_matries, reconstruct_matries = compute_norm(self.combine_matrix)
		return normed_matries, reconstruct_matries

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
			
			X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
			self.list_Ti.append(np.copy(X[0]))

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
		print(self.list_alpha)
		return self.list_alpha


	def interpolate_missing(self):
		list_matrix = []
		for j in range(self.K):
			predictAj = []
			for g in range(len(self.markerwithgap)):
				# list_matrix.append(self.F_matrix[g].interpolate_A[0])
				predictAj.append(self.F_matrix[g].interpolate_A[j])
			tmp = self.list_alpha[j] * summatrix_list(predictAj)
			list_matrix.append(tmp)
		result = summatrix_list(list_matrix)
		result = result[-self.missing_matrix.shape[0]:] + self.MeanMat[-self.missing_matrix.shape[0]:]

		final_result = np.copy(self.original_missing)
		final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
		return final_result


class interpolation_weighted_gap_dang_v5():

	def __init__(self, reference_matrix, missing_matrix, refine = False):
		self.fix_leng = missing_matrix.shape[0]
		self.combine_matrix = np.vstack((np.copy(reference_matrix), np.copy(missing_matrix)))
		self.missing_matrix = missing_matrix
		self.reference_matrix = np.copy(reference_matrix)
		self.original_missing = missing_matrix
		self.mean_error = -1
		counter = 0
		while refine:
			counter += 1
			if counter > 1: break
			current_mean_change = False
			current_number_patch = self.reference_matrix.shape[0] // missing_matrix.shape[0]
			self.F_matrix = self.prepare(remove_patches = True, current_mean = self.mean_error)
			self.list_frameidx_patch = get_list_frameidx_patch(self.K, self.fix_leng, 0)
			differ = (self.mean_error-self.min_error)*100/self.min_error
			if (differ < 6):
				self.reference_matrix = np.copy(reference_matrix)
				self.combine_matrix = np.vstack((np.copy(self.reference_matrix), np.copy(missing_matrix)))
				break
			if (differ > 6):
				tmp = np.mean(np.asarray(self.list_error)) + 0.0001
				self.mean_error =  tmp
				current_mean_change = True
			if (len(self.list_patches) == current_number_patch) and (current_mean_change == False):
				break
			if len(self.list_patches) == 0:
				self.list_patches = [x for x in range(reference_matrix.shape[0] // missing_matrix.shape[0])]
				self.reference_matrix = np.copy(reference_matrix)
				self.combine_matrix = np.vstack((np.copy(self.reference_matrix), np.copy(missing_matrix)))
				break	
			list_newData = []
			for i in self.list_patches:
				[l, r] = self.list_frameidx_patch[i]
				list_newData.append(self.combine_matrix[l:r])
			self.reference_matrix = np.vstack(list_newData)
			self.combine_matrix = np.vstack((np.copy(self.reference_matrix), np.copy(missing_matrix)))

		self.normed_matries, self.reconstruct_matries = self.normalization()
		self.A1 = np.copy(self.normed_matries[0])
		self.AN = np.copy(self.normed_matries[1])
		self.AN0 = np.copy(self.normed_matries[2])
		self.K = int(self.AN.shape[0] / self.fix_leng)
		self.compute_svd()
		self.F_matrix = self.prepare()

		self.result_norm = self.interpolate_missing()

	def prepare(self, remove_patches = False, current_mean = -1):
		list_F_matrix = []
		DistalThreshold = 0.5
		test_data = np.copy(self.missing_matrix)
		source_data = np.copy(self.reference_matrix)

		AA = np.copy(self.combine_matrix)
		columnindex = np.where(AA == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		# print(markerwithgap)
		self.markerwithgap = markerwithgap
		missing_frame_testdata = np.unique(np.where(test_data == 0)[0])
		list_frame = np.arange(test_data.shape[0])
		full_frame_testdata = np.asarray([i for i in list_frame if i not in missing_frame_testdata])
		self.full_frame_testdata = full_frame_testdata
		
		[frames, columns] = AA.shape
		Data_without_gap = np.delete(np.copy(AA), columnwithgap, 1)
		mean_data_withoutgap_vec = np.mean(Data_without_gap, 1).reshape(Data_without_gap.shape[0], 1)
		columnWithoutGap = Data_without_gap.shape[1]

		x_index = [x for x in range(0, columnWithoutGap, 3)]
		mean_data_withoutgap_vecX = np.mean(Data_without_gap[:,x_index], 1).reshape(frames, 1)

		y_index = [x for x in range(1, columnWithoutGap, 3)]
		mean_data_withoutgap_vecY = np.mean(Data_without_gap[:,y_index], 1).reshape(frames, 1)

		z_index = [x for x in range(2, columnWithoutGap, 3)]
		mean_data_withoutgap_vecZ = np.mean(Data_without_gap[:,z_index], 1).reshape(frames, 1)

		joint_meanXYZ = np.hstack((mean_data_withoutgap_vecX, mean_data_withoutgap_vecY, mean_data_withoutgap_vecZ))
		self.MeanMat = np.tile(joint_meanXYZ, AA.shape[1]//3)
		# np.savetxt("checkMeanmat.txt", self.MeanMat[-self.missing_matrix.shape[0]:], fmt = "%.2f")
		AA = AA - self.MeanMat
		AA[np.where(self.combine_matrix == 0)] = 0
		self.norm_Data = np.copy(AA)
		#  /////////////////////////////////////////////////////////////////////////
		# note recontruct Meanmat
		self.K = self.reference_matrix.shape[0] + len(full_frame_testdata)
		self.K = int(self.K / self.fix_leng)
		self.list_patches = [x for x in range(self.K)]
		self.list_mark_matrixQ = []
		for marker in markerwithgap:
			tmp = np.zeros(self.missing_matrix.shape)
			missing_frame_marker = np.where(self.missing_matrix[:, marker*3] == 0)
			tmp[missing_frame_marker, marker*3 : marker*3+3] = 1
			self.list_mark_matrixQ.append(np.copy(tmp))

		self.Q_matrix = np.zeros(self.missing_matrix.shape)
		self.Q_matrix[np.where(self.missing_matrix == 0)] = 1

		list_frameidx_patch = get_list_frameidx_patch(self.K, self.fix_leng, len(full_frame_testdata))
		self.list_frameidx_patch = list_frameidx_patch
		counter_marker = 0
		# print(markerwithgap)
		for marker in markerwithgap:
			# print(marker)
			missing_frame = np.where(test_data[:, marker*3] == 0)
			# print(missing_frame)
			EuclDist2Marker = compute_weight_vect_norm([marker], AA)
			thresh = np.mean(EuclDist2Marker) * DistalThreshold
			# print(thresh)
			# stop
			Data_remove_joint = np.copy(AA)
			for sub_marker in range(len(EuclDist2Marker)):
				if (EuclDist2Marker[sub_marker] > thresh) and (sub_marker in markerwithgap):
					Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			Data_remove_joint[:, marker*3:marker*3+3] = np.copy(AA[:, marker*3:marker*3+3])
			frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)[0]
			for sub_marker in markerwithgap:
				if sub_marker != marker:
					if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
						# print(marker, " affect ", sub_marker)
						Data_remove_joint[:,sub_marker*3:sub_marker*3+3] = 0
			
			source_data_renew = Data_remove_joint[: len(self.reference_matrix)]
			full_frame_renew = full_frame_testdata + self.reference_matrix.shape[0]
			full_joint_testingData = Data_remove_joint[full_frame_renew, :]
			only_missing_testingData = Data_remove_joint[frames_missing_marker, :]
			Data_renew =  np.vstack([source_data_renew, full_joint_testingData])
			full_testData = np.vstack([Data_renew, only_missing_testingData])
			# np.savetxt("checkinput.txt", Data_remove_joint, fmt = "%.2f")
			# stop
			if remove_patches:
				gap_interpolation = interpolation_gap_patch_v3(Data_renew, marker, self.missing_matrix, list_frameidx_patch, full_testData, Data_remove_joint, np.copy(AA))
			else:
				gap_interpolation = interpolation_gap_patch_v2(Data_renew, marker, self.missing_matrix, list_frameidx_patch, full_testData, Data_remove_joint, np.copy(AA))
			list_F_matrix.append(gap_interpolation)
			counter_marker += 1
		if remove_patches:
		# remove patches having distal with testing data
			list_error = []
			for i in range(self.K):
				tmp = 0
				for g in range(len(markerwithgap)):
					tmp += list_F_matrix[g].list_error_patch[i]
				list_error.append(tmp)
			self.list_error = list_error
			min_error =np.min(np.asarray(list_error))
			self.min_error = min_error
			if current_mean < 0:
				mean_error = np.mean(np.asarray(list_error))
				self.mean_error = mean_error + 0.000000001

			print(list_error)
			print(self.mean_error)
			# self.mean_error = max(self.mean_error, 4)
			list_patches = []
			for i in range(self.K):
				if list_error[i] <= self.mean_error:
					list_patches.append(i)
			print(list_patches)
			
			self.list_patches = list_patches
		return list_F_matrix

	def normalization(self):
		normed_matries, reconstruct_matries = compute_norm(self.combine_matrix)
		return normed_matries, reconstruct_matries

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
			
			X = np.linalg.lstsq(Ai0UN0, AiUN, rcond = None)
			self.list_Ti.append(np.copy(X[0]))

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
		print(self.list_alpha)
		return self.list_alpha


	def interpolate_missing(self):
		list_matrix = []
		for j in range(self.K):
			predictAj = []
			patch_in_list = self.list_patches[j]
			for g in range(len(self.markerwithgap)):
				# list_matrix.append(self.F_matrix[g].interpolate_A[1])
				predictAj.append(self.F_matrix[g].interpolate_A[patch_in_list])
			tmp = self.list_alpha[patch_in_list] * summatrix_list(predictAj)
			list_matrix.append(tmp)
		result = summatrix_list(list_matrix)
		result = result[-self.missing_matrix.shape[0]:] + self.MeanMat[-self.missing_matrix.shape[0]:]

		final_result = np.copy(self.original_missing)
		final_result[np.where(self.original_missing == 0)] = result[np.where(self.original_missing == 0)]
		return final_result
