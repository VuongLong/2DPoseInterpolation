import numpy as np
import argparse
import sys
import cv2
import os


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [0, 255, 255], \
		  [85, 0, 255], [0, 255, 0], [255, 0, 170], [255, 0, 0], [0, 255, 255], \
		  [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 255, 0], [170, 255, 0], \
		  [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0], [255, 85, 0], \
		  [255, 170, 0], [255, 255, 0], [170, 255, 0],[0, 255, 255], [0, 170, 255]]

pose = [(0, 1), (1, 2), (1, 8), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (0, 15), 
		(0, 16), (15, 17), (16, 18), (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), 
		(22, 23), (11, 24), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]

draw_colors = [0, 1, 14, 3, 4, 5, 3, 4, 8, 9, 10, 11, 12, 11, 0, 15, 22, 23, 18, 0, 15, 18, 22, 23]


def drawline(img,a,b,Xs,Ys,c):
	 cv2.line(img, (int(Xs[a]), int(Ys[a])), (int(Xs[b]), int(Ys[b])), colors[c], 2)  


def contruct_skeletion_to_image(fullfilenames, Tracking2D):
	img = cv2.imread(fullfilenames)
	
	Xs=[]
	Ys=[]
	for j in range(Tracking2D.shape[0]):
		x=round(float(Tracking2D[j][0]))
		y=round(float(Tracking2D[j][1]))
		Xs.append(x)
		Ys.append(y)
		cv2.circle(img,(x,y),2,(0,0,0),thickness=-1)

	for (joint, color) in zip(pose, draw_colors):
		drawline(img, joint[0], joint[1], Xs, Ys, color)

	return img


def contruct_skeletion_to_video(input_dir, Tracking2D, target, output_dir, output_video='output', ingore_confidence=False):
	print(input_dir + '/images')
	files = os.listdir(input_dir + '/images')
	print(files.sort()) 
	image_file = input_dir + '/images/' + files[0]
	img = cv2.imread(image_file)
	video = cv2.VideoWriter(output_dir + '/' + output_video + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (img.shape[1], img.shape[0]))
	
	joint_dim = 3
	if ingore_confidence:
		joint_dim = 2
	Tracking2D=Tracking2D.ravel().reshape(Tracking2D.shape[0],25,joint_dim)
	for i in range(target[0], target[1]):
		image_file = input_dir + '/images/' + files[i]
		print(image_file)
		img = contruct_skeletion_to_image(image_file, Tracking2D[i--target[0]])

		out_img = output_dir +'/'+files[i]
		cv2.imwrite(out_img, img)
		video.write(img)
	video.release()


def show_video(video_dir, wait_key=25):
	cap = cv2.VideoCapture(video_dir + '.avi')
	while(True):
		ret,frame = cap.read()#capture one frame
		if ret:
			cv2.imshow('frame', frame)
			cv2.waitKey(wait_key)
		else:
			break 
	cap.release()
	cv2.destroyAllWindows()









