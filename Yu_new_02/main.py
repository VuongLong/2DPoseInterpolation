import numpy as np
from Yu_new_02.preprocess import generate_patch_data, normalize
# from adaboost import *
from Yu_new_02.algorithm2 import * 
from Yu_new_02.utils import *

def interpolation_weighted_T_1702(source_data, test_data):
	interpolation = interpolation_weighted_T(source_data, test_data)
	result = interpolation.interpolate_missing()
	return result


def interpolation_T_1702(source_data, test_data):
	interpolation = Interpolation_T(source_data, test_data)
	result = interpolation.interpolate_missing()
	return result

if __name__ == '__main__':
	print("ok")