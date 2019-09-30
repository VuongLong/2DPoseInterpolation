import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random


link = ["./data3D/original_matrix.txt", "./data3D/recovered_PMA_matrix.txt"]
source1 , _  = read_tracking_data3D_v2(link[0])
source1 = source1.astype(float)
print(source1.shape)

source2 , _  = read_tracking_data3D_v2(link[1])
source2 = source2.astype(float)
print(source2.shape)

compare = source2 - source1
print(np.sqrt(np.sum(np.square(compare))/360))

# print(np.sum(np.abs(source2 - source1)))

np.savetxt("compare.txt",source2 - source1, fmt = "%.4f")