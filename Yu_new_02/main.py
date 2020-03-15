import numpy as np
from Yu_new_02.preprocess import generate_patch_data, normalize
# from adaboost import *
from Yu_new_02.algorithm2 import * 
from Yu_new_02.utils import *

def check_vector_overlapping(vector_check):
	if len(np.where(vector_check == 0)) > 0:
		return False
	return True

def interpolation_weighted_T_1702(source_data, test_data, norm = False, downsample = False, strategy_R2= False):
	DistalThreshold = 0.5
	if strategy_R2:
		result = np.copy(test_data)
		print("interpolation_weighted_T_1702 R2")
		AA = np.vstack((np.copy(source_data), np.copy(test_data)))
		columnindex = np.where(AA == 0)[1]
		columnwithgap = np.unique(columnindex)
		markerwithgap = np.unique(columnwithgap // 3)
		frameindex = np.where(AA == 0)[0]
		framewithgap = np.unique(frameindex)
		for marker in markerwithgap:
			missing_frame = np.where(test_data[:, marker*3] == 0)
			EuclDist2Marker = compute_weight_vect_norm([marker], downsample, AA)
			thresh = np.mean(EuclDist2Marker) * DistalThreshold
			# remove the columns distal to current marker
			Data_remove_joint = np.copy(AA)
			for sub_marker in range(len(EuclDist2Marker)):
				if EuclDist2Marker[sub_marker] > thresh:
					Data_remove_joint[:,sub_marker*3:marker*3+3] = 0
			Data_remove_joint[:, marker*3:marker*3+3] = AA[:, marker*3:marker*3+3]
			
			frames_missing_marker = np.where(Data_remove_joint[:, marker*3] == 0)
			# find overlap gap with marker
			for sub_marker in markerwithgap:
				if sub_marker != marker:
					if check_vector_overlapping(Data_remove_joint[frames_missing_marker, sub_marker*3]):
						Data_remove_joint[:,sub_marker*3:marker*3+3] = 0
			

			# remove frames with incomplete information
			len_sourcedata = source_data.shape[0]
			partly_sourcedata = Data_remove_joint[:len_sourcedata,:]
			# add frames with gaps in marker into the end of the matrix
			# change to our algorithm, that would be test_data
			len_testdata = test_data.shape[0]
			partly_test_data = Data_remove_joint[-len_testdata:,:]

			interpolation_partly = interpolation_weighted_T(partly_sourcedata, partly_test_data, norm, downsample, strategy_R2 = True, marker = marker)
			result[missing_frame, marker*3 : marker*3+3]  = interpolation_partly.result_norm[missing_frame, marker*3 : marker*3+3]
	else:
		interpolation = interpolation_weighted_T(source_data, test_data, norm, downsample)
		if norm:
			result = interpolation.result_norm
		else:
			result = interpolation.result_nonorm
	return result


def interpolation_T_1702(source_data, test_data, norm = False, strategy_R2= False):
	interpolation = Interpolation_T(source_data, test_data, norm)
	if norm:
		result = interpolation.result_norm
	else:
		result = interpolation.result_nonorm
	return result

if __name__ == '__main__':
	print("ok")