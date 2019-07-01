import numpy as np
import argparse
import sys
import cv2
import os
import xlwt 
from xlwt import Workbook
import matplotlib.pyplot as plt


colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [0, 255, 255], \
		  [85, 0, 255], [0, 255, 0], [255, 0, 170], [255, 0, 0], [0, 255, 255], \
		  [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 255, 0], [170, 255, 0], \
		  [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0], [255, 85, 0], \
		  [255, 170, 0], [255, 255, 0], [170, 255, 0],[0, 255, 255], [0, 170, 255]]

pose = [(0, 1), (1, 2), (1, 8), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (0, 15), 
		(0, 16), (15, 17), (16, 18), (8, 9), (8, 12), (9, 10), (10, 11), (11, 22), 
		(22, 23), (11, 24), (12, 13), (13, 14), (14, 21), (14, 19), (19, 20)]

draw_colors = [0, 1, 14, 3, 4, 5, 3, 4, 8, 9, 10, 11, 12, 11, 0, 15, 22, 23, 18, 0, 15, 18, 22, 23]

color_plot = ['g', 'r', 'y', 'b', 'k' , 'c']


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
	files.sort()
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


def show_video(video_dir, wait_key=100):
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


def export_xls(M1_result1, M1_result2, M1_result3 = None, M2_result1 = None, M2_result2 = None, file_name="default"):
	# Workbook is created 
	wb = Workbook() 
	tmp = np.array(M1_result1).shape
	# add_sheet is used to create sheet. 
	sheet1 = wb.add_sheet('Sheet 1') 
	space = 2
	if M1_result3 != None:
		space = 3
	for x in range(tmp[1]):
		for y in range(tmp[0]):
			sheet1.write(x, y*space, M1_result1[y][x]) 
			sheet1.write(x, y*space+1, M1_result2[y][x]) 
			if M1_result3 != None:
				sheet1.write(x, y*space+2, M1_result3[y][x])

	if (M2_result1 != None):
		for x in range(tmp[1]):
			for y in range(tmp[0]):
				sheet1.write(x+10, y*2, M2_result1[y][x]) 
				sheet1.write(x+10, y*2+1, M2_result2[y][x]) 

	wb.save(str(file_name)+'.xls')

def plot_line(M1_result1, M1_result2, title, type = "joint", name1 = "Error A0", name2 = "Error A1", scale = "1"):
	fig = plt.figure(figsize=(20,10))
	fig.suptitle(title, fontsize=10)
	tmp = np.copy(np.array(M1_result1).T)
	plt.subplot(211)
	for idx, x in enumerate(tmp):
		yy = np.arange(x.shape[0])
		plt.plot(yy*scale, x, color = color_plot[idx], marker = '.', linewidth=2.0, label="Number missing"+type+' '+str(idx+1))
	plt.legend(loc = 0, mode="expand", ncol= 2)
	plt.ylabel(name1)
	print(plt.ylim())
	plt.ylim((0, 6))


	tmp = np.copy(np.array(M1_result2).T)
	plt.subplot(212)
	for idx, x in enumerate(tmp):
		yy = np.arange(x.shape[0])
		plt.plot(yy*scale, x, color = color_plot[idx], marker = '.', linewidth=2.0)
	plt.xlabel('Number shifted frame')
	plt.ylabel(name2)
	print(plt.ylim())
	plt.ylim((0, 6))
	plt.show()
	fig.savefig(title+'.jpg')


def plot_line3(M1_result1, M1_result2, M1_result3, title, type = "joint", name1 = "Error A0", name2 = "Error A1", scale = "1"):
	fig = plt.figure(figsize=(20,10))
	fig.suptitle(title, fontsize=10)
	tmp = np.copy(np.array(M1_result1).T)
	plt.subplot(311)
	for idx, x in enumerate(tmp):
		yy = np.arange(x.shape[0])
		plt.plot(yy*scale, x, color = color_plot[idx], marker = '.', linewidth=2.0, label="Number missing"+type+' '+'5')
	plt.legend(loc = 0, mode="expand", ncol= 2)
	plt.ylabel("Error of Task 5")
	print(plt.ylim())
	plt.ylim((0, 15))


	tmp = np.copy(np.array(M1_result2).T)
	plt.subplot(312)
	for idx, x in enumerate(tmp):
		yy = np.arange(x.shape[0])
		plt.plot(yy*scale, x, color = color_plot[idx], marker = '.', linewidth=2.0)
	plt.ylabel("Error of Task 3")
	print(plt.ylim())
	plt.ylim((0, 15))

	tmp = np.copy(np.array(M1_result3).T)
	plt.subplot(313)
	for idx, x in enumerate(tmp):
		yy = np.arange(x.shape[0])
		plt.plot(yy*scale, x, color = color_plot[idx], marker = '.', linewidth=2.0)
	plt.xlabel('Number shifted frame')
	plt.ylabel("Error of Task 4")
	print(plt.ylim())
	plt.ylim((0, 15))
	plt.show()
	fig.savefig(title+'.jpg')
