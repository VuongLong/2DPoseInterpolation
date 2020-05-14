# implemention corresponds 1st formula to check result
import numpy as np
from data_utils import *
from render_utils import *
from algorithm import *
from arguments import arg
import sys
import random
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec

def draw(ax1, ax2, ax3, link_data, joint_index, list_frame, color_set, axins1, axins2, axins3):
	Tracking3D, _  = read_tracking_data3D_v3(link_data)
	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D.T

	x1 = joint_index*3
	x2 = joint_index*3+1
	x3 = joint_index*3+2
	xx = Tracking3D[x1]
	yy = Tracking3D[x2]
	zz = Tracking3D[x3]


	ax1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	axins1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	ax2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	axins2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	ax3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)
	axins3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], color=color_set, linewidth=1)

if __name__ == '__main__':
	color = ['g', 'r', 'y', 'b', 'k' , 'c']

	
	data_link = ["./missing_matrix.txt"]
	Tracking3D, _  = read_tracking_data3D_v3(data_link[0])
	Tracking3D = Tracking3D.astype(float)
	tmp = np.where(Tracking3D == 0)[0]
	list_frame = np.unique(tmp)
	ox = [x for x in range(Tracking3D.shape[0])]
	Tracking3D = Tracking3D.T



	data_link = ["./original.txt"]
	Tracking3D, _  = read_tracking_data3D_v3(data_link[0])
	Tracking3D = Tracking3D.astype(float)
	Tracking3D = Tracking3D.T
	number_joint = Tracking3D.shape[0] // 3
	




	title = "Joint curve"
	fig = plt.figure()
	# fig.suptitle(title, fontsize=16)
	joint_index = 3
	x1 = joint_index*3
	x2 = joint_index*3+1
	x3 = joint_index*3+2
	xx = Tracking3D[x1]
	yy = Tracking3D[x2]
	zz = Tracking3D[x3]
	
	gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1]) 
	ax1 = plt.subplot(gs[0])
	ax1.set_ylabel(' x axis ')
	ax1.plot(ox[:list_frame[0]+1], xx[:list_frame[0]+1], color='g', linewidth=1)
	ax1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], linestyle="--",color='g', linewidth=1)
	ax1.plot(ox[list_frame[-1]-1:], xx[list_frame[-1]-1:], color='g', linewidth=1)
	axins1 = zoomed_inset_axes(ax1, 1.5, loc=10,  bbox_to_anchor=(3000,3300))
	axins1.plot(ox[list_frame[0]:list_frame[-1]], xx[list_frame[0]:list_frame[-1]], linestyle="--",color='g', linewidth=1)
	x1, x2, y1, y2 =  48, 73, -97.4, -94
	axins1.set_xlim(x1, x2) # apply the x-limits
	axins1.set_ylim(y1, y2) # apply the y-limits
	plt.yticks(visible=False)
	plt.xticks(visible=False)
	mark_inset(ax1, axins1, loc1=1, loc2=3, fc="none", ec="0.5")


	ax2 = plt.subplot(gs[2])
	ax2.set_ylabel(' y axis ')
	ax2.plot(ox[:list_frame[0]+1], yy[:list_frame[0]+1], color='r', linewidth=1)
	ax2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], linestyle="--",color='r', linewidth=1)
	ax2.plot(ox[list_frame[-1]-1:], yy[list_frame[-1]-1:], color='r', linewidth=1)

	axins2 = zoomed_inset_axes(ax2, 1.5, loc=10,  bbox_to_anchor=(3000, 2400))
	axins2.plot(ox[list_frame[0]:list_frame[-1]], yy[list_frame[0]:list_frame[-1]], linestyle="--",color='r', linewidth=1)
	x1, x2, y1, y2 =  48, 73, 93.25, 93.30
	axins2.set_xlim(x1, x2) # apply the x-limits
	axins2.set_ylim(y1, y2) # apply the y-limits
	plt.yticks(visible=False)
	plt.xticks(visible=False)
	mark_inset(ax2, axins2 , loc1=1, loc2=3, fc="none", ec="0.5")

	ax3 = plt.subplot(gs[4])
	ax3.set_ylabel(' z axis ')
	ax3.plot(ox[:list_frame[0]+1], zz[:list_frame[0]+1], color='y', linewidth=1)
	ax3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], linestyle="--",color='y', linewidth=1)
	ax3.plot(ox[list_frame[-1]-1:], zz[list_frame[-1]-1:], color='y', linewidth=1)

	axins3 = zoomed_inset_axes(ax3, 1.5, loc=10,  bbox_to_anchor=(3000, 1200))
	axins3.plot(ox[list_frame[0]:list_frame[-1]], zz[list_frame[0]:list_frame[-1]], linestyle="--",color='y', linewidth=1)
	x1, x2, y1, y2 =  48, 73, 286, 290
	axins3.set_xlim(x1, x2) # apply the x-limits
	axins3.set_ylim(y1, y2) # apply the y-limits
	plt.yticks(visible=False)
	plt.xticks(visible=False)
	mark_inset(ax3, axins3 , loc1=1, loc2=3, fc="none", ec="0.5")

	ax3.set_xlabel('Frames')

	# done original

	data_link = "./interpolate.txt"
	draw(ax1, ax2, ax3, data_link, joint_index, list_frame, "b", axins1, axins2, axins3)
	
	fig.set_size_inches(6, 8)
	
	
	
	# plt.show()
	plt.savefig('./output/'+title+'.png', dpi=600)
