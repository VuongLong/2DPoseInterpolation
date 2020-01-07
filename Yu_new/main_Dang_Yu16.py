import numpy as np
from Yu_new.preprocess import generate_patch_data, normalize
# from adaboost import *
from Yu_new.algorithm1 import * 
from Yu_new.utils import *
import copy

class adaboost_16th():
	def __init__(self, inner_function, number_loop = 20):
		self.iteration_lim = number_loop
		self.function = inner_function
		self.list_function = []
		self.list_function.append(self.function)
		self.list_beta = []
		self.threshold = 0.8
		self.power_coefficient = 10
		self.number_sample = inner_function.get_number_sample()
		self.limit_error = 2
		self.list_mean_error = []

	def set_iteration(self, number):
		self.iteration_lim = number

	def train(self):
		index_maxError = -1
		index_maxPosition = -1
		for loop_i in range(self.iteration_lim):
			print("looping: ", loop_i)
			current_function = self.list_function[-1]
			accumulate_error_weight = 0
			
			# compute error for each sample
			error_sample = current_function.interpolate_sample()
			self.list_mean_error.append(np.mean(error_sample))
			for x in range(len(error_sample)):
				if error_sample[x] > error_sample[index_maxError]:
					index_maxError = x
				if (error_sample[index_maxError] > self.limit_error) and (index_maxPosition == -1):
					index_maxPosition = loop_i
			
			self.threshold = np.median(error_sample)
			print(error_sample)
			print("threshold: ", self.threshold)
			alpha = current_function.get_alpha()
			weight_sample = current_function.get_weight()
			if (loop_i - index_maxPosition >= 10) and (index_maxPosition != -1):
				weight_sample[index_maxError] = 0
				index_maxPosition = -1
			print("weight_sample", weight_sample)
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
			print("accumulate_Z: ", accumulate_Z)
			print("accumulate_error_weight", accumulate_error_weight)
			if accumulate_error_weight <= 0.00001:
				self.iteration_lim = loop_i-1
				print("stop training ADABOOST")
				break
			if accumulate_error_weight >= 0.9999:
				self.iteration_lim = loop_i
				print("stop training ADABOOST")
				break
			for x in range(self.number_sample):
				new_distribution[x] = new_distribution[x] / accumulate_Z
			print(new_distribution)
			print("ok")
			# update new function
			new_function = copy.deepcopy(current_function)
			new_function.set_weight(np.copy(new_distribution))
			new_function.inner_compute_alpha()
			self.list_function.append(new_function)

	def get_beta_info(self):
		return self.list_beta

	def interpolate_accumulate(self):
		if self.iteration_lim < 1:
			return self.function.interpolate_missing()
		list_result = []
		start_round = 0
		if self.iteration_lim >= 2:
			for x in range(min(self.iteration_lim-1, 2)):
				if (self.list_mean_error[x] < np.mean(np.asarray(self.list_mean_error[x+1:self.iteration_lim]))):
					break
				start_round = x+1
		# if (self.list_mean_error[0] > np.mean(np.asarray(self.list_mean_error[1:-1]))) and self.iteration_lim >= 2:
		# 	start_round = 1
		# 	print("///////////////////////////////////////////")
		# 	print("bi tru")
		# 	print("///////////////////////////////////////////")
		for t in range(start_round, self.iteration_lim):
			current_function = self.list_function[t]
			function_result = current_function.interpolate_missing()
			list_result.append(function_result)
		result = np.zeros(list_result[-1].shape)
		sum_beta = 0
		counter = 0
		for t in range(start_round, self.iteration_lim):
			function_beta = np.log(1/self.list_beta[t])
			result += function_beta * list_result[counter]
			counter+= 1 
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
	interpolation = Interpolation16th_F(source_data, test_data)
	interpolation.interpolate_missing()
	boosting = adaboost_16th(interpolation) 
	boosting.train()
	print(boosting.get_beta_info())
	print(boosting.get_arbitrary_sample().get_alpha())
	result = boosting.interpolate_accumulate()
	return result

if __name__ == '__main__':
	print("ok")
	