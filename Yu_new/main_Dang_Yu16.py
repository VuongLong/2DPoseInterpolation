import numpy as np
from Yu_new.preprocess import generate_patch_data, normalize
# from adaboost import *
from Yu_new.algorithm1 import * 
from Yu_new.utils import *
from Yu_new.data import AN_T, AN_F, A1, original_A1, missing_map
import copy

class adaboost_16th():
	def __init__(self, inner_function, number_loop = 40):
		self.iteration_lim = number_loop
		self.function = inner_function
		self.list_function = []
		self.list_function.append(self.function)
		self.list_beta = []
		self.threshold = 0.6
		self.power_coefficient = 2
		self.number_sample = inner_function.get_number_sample()
	def set_iteration(self, number):
		self.iteration_lim = number

	def train(self):
		for loop_i in range(self.iteration_lim):
			current_function = self.list_function[-1]
			accumulate_error_weight = 0
			
			# compute error for each sample
			error_sample = current_function.interpolate_sample()
			alpha = current_function.get_alpha()
			weight_sample = current_function.get_weight()
			# compute error rate for function
			for x in range(self.number_sample):
				if error_sample[x] > self.threshold:
					accumulate_error_weight += weight_sample[x]
			current_beta = accumulate_error_weight ** self.power_coefficient
			self.list_beta.append(current_beta)
			new_distribution = []
			accumulate_Z = 0
			for x in range(self.number_sample):
				if error_sample[x] <= self.threshold:
					new_distribution.append(weight_sample[x] * current_beta)
				else:
					new_distribution.append(weight_sample[x])
				accumulate_Z += new_distribution[-1]
			for x in range(self.number_sample):
				new_distribution[x] = new_distribution[x] / accumulate_Z
			# update new function
			new_function = copy.deepcopy(current_function)
			new_function.set_weight(np.copy(new_distribution))
			new_function.inner_compute_alpha()
			self.list_function.append(new_function)

	def get_beta_info(self):
		return self.list_beta

	def interpolate_accumulate(self):
		list_result = []
		for t in range(self.iteration_lim):
			current_function = self.list_function[t]
			function_result = current_function.interpolate_missing()
			list_result.append(function_result)
		result = np.zeros(list_result[-1].shape)
		sum_beta = 0
		for t in range(self.iteration_lim):
			function_beta = np.log(1/self.list_beta[t])
			result += function_beta * list_result[t] 
			sum_beta += function_beta
		final_result = result / sum_beta
		return final_result


	def get_distribution_sample(self):
		list_distri = []
		for function in self.list_function:
			list_distri.append(function.get_weight())
		return	list_distri


	def get_arbitrary_sample(self, sample_idx = -1):
		return self.list_function[sample_idx]

def test_func(source_data, test_data):
	interpolation = Interpolation16th_F(AN_F, test_data)
	interpolation.interpolate_missing()
	boosting = adaboost_16th(interpolation) 
	boosting.train()
	print(boosting.get_beta_info())
	print(boosting.get_arbitrary_sample().get_alpha())
	result = boosting.interpolate_accumulate()
	return result

if __name__ == '__main__':

	print("Reference source:")
	print('AN_T', AN_T.shape)
	print('AN_F', AN_F.shape)

	print("\nTest source:")
	print('A1', A1.shape)
	
	interpolation = Interpolation16th_F(AN_F, A1)
	result1 = interpolation.interpolate_missing()
	print(MSE(result1, original_A1, missing_map))
	boosting = adaboost_16th(interpolation) 
	boosting.train()
	print(boosting.get_beta_info())
	result2 = boosting.interpolate_accumulate()
	print(MSE(result2, original_A1, missing_map))
