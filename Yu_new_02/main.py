import numpy as np
from Yu_new_02.preprocess import generate_patch_data, normalize
# from adaboost import *
from Yu_new_02.algorithm2 import * 
from Yu_new_02.utils import *

def interpolation_weighted_T_1702(source_data, test_data, norm = False):
	interpolation = interpolation_weighted_T(source_data, test_data, norm)
	if norm:
		result = interpolation.result_norm
	else:
		result = interpolation.result_nonorm
	return result


def interpolation_T_1702(source_data, test_data, norm = False):
	interpolation = Interpolation_T(source_data, test_data, norm)
	if norm:
		result = interpolation.result_norm
	else:
		result = interpolation.result_nonorm
	return result

def interpolation_weighted_T_gap(source_data, test_data):
	interpolation = interpolation_weighted_gap(source_data, test_data)
	result = interpolation.result_norm
	return result



if __name__ == '__main__':
	print("ok")