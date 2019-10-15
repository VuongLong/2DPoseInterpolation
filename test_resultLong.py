import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


link = ["./result_checking/135_02.txt", "./result_checking/recovered_PMA_1.txt", "./result_checking/1.txt"]
source1 , _  = read_tracking_data3D_v2(link[0])
source1 = source1.astype(float)

source2 , _  = read_tracking_data3D_v2(link[1])
source2 = source2.astype(float)


mask , _  = read_tracking_data3D_v3(link[2])
mask = mask.astype(int)

origin = source1[:2800]

print(np.around(calculate_mae_matrix( origin[np.where(mask == 0)]- source2[np.where(mask == 0)]), decimals = 17))
print(np.around(calculate_mse_matrix( origin[np.where(mask == 0)]- source2[np.where(mask == 0)]), decimals = 17))
halt
print(np.sqrt(np.sum(np.square(compare))/360))

# print(np.sum(np.abs(source2 - source1)))

