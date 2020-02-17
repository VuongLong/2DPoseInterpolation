import numpy as np
from Yu_new_02.preprocess import generate_patch_data, normalize
# from adaboost import *
from Yu_new_02.algorithm_Yu2 import * 
from Yu_new_02.utils import *

def interpolation_1002(source_data, test_data):
	interpolation = Interpolation16th_F(source_data, test_data)
	interpolation.interpolate_missing()
	result = interpolation.interpolate_missing()
	return result

if __name__ == '__main__':
	print("ok")
	