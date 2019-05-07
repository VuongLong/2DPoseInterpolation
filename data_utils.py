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
