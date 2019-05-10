import numpy as np


def read_tracking_data(data_dir, ingore_confidence=False):
	Tracking2D = []
	f=open(data_dir, 'r')
	j=0
	for line in f:
		Tracking2D.append([])
		elements = line.split(',')
		Tracking2D[j].append( [elements[i] for i in range(len(elements))] )
		j+=1
	f.close()
	Tracking2D = np.array(Tracking2D) # list can not read by index while arr can be
	Tracking2D = np.squeeze(Tracking2D)

	if ingore_confidence:
		Tracking2D = Tracking2D.reshape(Tracking2D.shape[0],25,3)
		Tracking2D = Tracking2D[..., 0:2]
		Tracking2D = Tracking2D.reshape(Tracking2D.shape[0],50)
	return Tracking2D

def find_full_matrix(Tracking2D, frame_length, overlap=False):
	count = 0
	list_full = []
	for frame in Tracking2D:
		if frame[frame==0].shape[0] == 0:
			count++
			if count > frame_length:
				list_full.append([count-frame_length, count])
				if not overlap:
					count = 0
	return list_full

# Todo
def find_miss_matrix(Tracking2D, frame_length, bum_frame_miss, max_miss_in_a_frame):
	pass

# Todo
# plot miss_matrix on video to check
def contruct_sellect_matrix()
	pass
