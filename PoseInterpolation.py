import numpy as np
from utils import *
from algorithm import *
from arguments import arg

if __name__ == '__main__':

    Tracking2D  = read_tracking_data(arg.data_dir, arg.ingore_confidence)
    

    A = Tracking2D[arg.reference[0]:arg.reference[1]].astype(float)
    A1 = Tracking2D[arg.target[0]:arg.target[1]].astype(float)

    A_star = interpolation(A,A1)
    
    #random_drop_joint(A)
    contruct_skeletion_to_video(arg.input_dir, A1, arg.target, arg.output_dir, arg.output_video, arg.ingore_confidence)
    
    show_video(arg.output_dir + '/' + arg.output_video, 200)
