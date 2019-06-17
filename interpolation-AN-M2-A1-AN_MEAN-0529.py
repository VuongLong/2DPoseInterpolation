# -*- coding: utf-8 -*-
"""
IMPORTANT: all the means are basing on the AN' mean

@author: Zhuorong Li
"""

import numpy as np
import cv2
import os

def read_txt(filename):
    f = open(filename)
    lines = f.readlines()      
    #X = np.zeros((70,75),dtype=float)
    ttrow = len(lines)
    #print(ttrow)
    ttcol = 75
    X = np.zeros((ttrow,ttcol),dtype=float)
    X_row = 0            
    for line in lines:       
      list = line.strip('\n').split(',')   
      X[X_row:] = list[0:75]          
      X_row+=1               
    #print(X)
    return X

'''SVD'''
def mysvd(dataMat):
    U, Sigma, VT = np.linalg.svd(dataMat)
    #print(Sigma) # Sigma is a row vector
    #Sigma_mat = np.mat(np.eye(75) * Sigma[:75]) # change Sigma to a matrix 
    #print(Sigma_mat)
    return U

'''function that replaces the zero entities in A1 with A*'''    
def replaceZeros(dataMat1,dataMat2):
    """
    dataMat1: the deficiency matrix A1
    dataMat2: A*
    return newMat that shares the non-zero entities of A1, 
                  while the zeroes are replaced with the corresponding element in A*
    """
    newMat = dataMat1
    for i in range(len(dataMat1)):
        for j in range(len(dataMat1[i])):
            if dataMat1[i][j]==0:
                newMat[i][j] = dataMat2[i][j]
                #print([i,j])
    return newMat

def findzeros(Mat):
    for i in range(len(Mat)):
        for j in range(len(Mat[i])):
            if Mat[i][j]==0:
                #print([i,j],Mat[i][j])
                print([i,j])
                
                
def setzeros(A0,A1):
    for i in range(len(A1)):
        for j in range(len(A1[i])):
            if A1[i][j] == 0:
                A0[i][j] = 0
    return A0

def findandsaveindex(Mat):
    list = []
    for i in range(len(Mat)):
        #print(i)
        for j in range(len(Mat[i])):
            if Mat[i][j]==0:
                list.append([i,j]) 
                #print([i,j],Mat[i][j])
                #print([i,j])
    arr = np.array(list)
    #arr = arr.astype(int)
    #np.savetxt('indexes.txt',arr)
    return arr

def interpolation(A,A1):
    A_MeanVec = np.mean(A,0)
    A_MeanMat = np.tile(A_MeanVec,(A.shape[0],1))
    A_new = A-A_MeanMat
    
    A1_MeanVec = A1.sum(0) / (A1 != 0).sum(0)
    #colMean = a.sum(0) / (a != 0).sum(0)
    #rowMean = a.sum(1) / (a != 0).sum(1)  
    A1_MeanMat = np.tile(A1_MeanVec,(A1.shape[0],1))
    A1_new = A1-A1_MeanMat
    A1_new = setzeros(A1_new,A1)
    
    A0 = A
    A0_new = A0-A_MeanMat
    A0_new = setzeros(A0_new,A1)
    
    '''compute U'''
    U = mysvd(np.matmul(A_new.T,A_new)) 
    #S = np.matmul(U,U.T)
        
    '''compute U0'''
    U0 = mysvd(np.matmul(A0_new.T,A0_new)) 
    #S0 = np.matmul(U0,U0.T)
    
    '''compute TMat'''
    TMat = np.matmul(U0.T,U)  #U = U0TMat
        
    '''compute U1'''
    U1 = mysvd(np.matmul(A1_new.T,A1_new)) 
    #S1 = np.matmul(U1,U1.T)
    
    '''compute A*'''
    Astar =  np.matmul(np.matmul(np.matmul(A1_new,U1),TMat),U.T)
    Astar = Astar + A1_MeanMat
    #print('finding zeros in Astar')
    findzeros(Astar)#no zero in Astar
    
    '''replace zero entities'''
    newMat = replaceZeros(A1,Astar)
    return newMat

def drawline(img,a,b,Xs,Ys,c):
	 cv2.line(img, (Xs[a], Ys[a]), (Xs[b], Ys[b]), colors[c], 2)    
     
def mse(reference, query):
    """Computes the Mean Square Error (MSE) 
    Parameters
    ----------
    reference: original data.
    query    : data to be compared.
    
    Return
    ----------
    value    : MSE value
    """
    (ref, que) = (reference.astype('double'), query.astype('double'))
    diff = ref - que
    square = (diff ** 2)
    mean = square.mean()
    return mean

        
if __name__ == '__main__':
    
    '''import the Tracking2Ddata'''
    #imgSeq = 'FastSong_Man_5'
    directory = 'E:/codes/BU/Mycodes/motion/FastSong_Man_5'
    save_directory = 'E:/codes/BU/Mycodes/motion/output'
    #directory = os.path.join('E:/codes/BU/Mycodes/motion/',imgSeq)
    #save_directory = os.path.join('E:/codes/BU/Mycodes/motion/output/',imgSeq)
    files = os.listdir(directory)
    Tracking2D = []
    fullfilename = directory+'/'+files[0]
    #videoName = imgSeq+'.avi'
    
    fps = 30 
    img1= cv2.imread(fullfilename)
    size = (img1.shape[1], img1.shape[0]) 
    video = cv2.VideoWriter("003.avi", cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    #video = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)

    #f=open('E:/codes/BU/Mycodes/motion/9_Fast Song 05.data', 'r')
    f=open('E:/codes/BU/Mycodes/motion/outputman_cut_1.data', 'r')
    j=0
    for line in f:
        Tracking2D.append([])
        elements = line.split(',')
        Tracking2D[j].append( [elements[i] for i in range(len(elements))] )
        j+=1
    f.close()
    
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [0, 255, 255], \
              [85, 0, 255], [0, 255, 0], [255, 0, 170], [255, 0, 0], [0, 255, 255], \
              [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 255, 0], [170, 255, 0], \
              [255, 0, 255], [255, 0, 170], [255, 0, 85],[255, 0, 0], [255, 85, 0], \
              [255, 170, 0], [255, 255, 0], [170, 255, 0],[0, 255, 255], [0, 170, 255]]
    
    Tracking2D = np.array(Tracking2D) # list can not read by index while arr can be
    Tracking2D = np.squeeze(Tracking2D) 
    zeroidx = findandsaveindex(Tracking2D.astype(float))
    AN = Tracking2D[0:600].astype(float)
    
    
    AN_MeanVec = np.mean(AN,0)
    N = 12
    m = 50
    AN_MeanMat_unit = np.tile(AN_MeanVec,(m,1)).T#75X50
    AN_MeanMat = np.tile(AN_MeanMat_unit,(N,1))#900X50
    AN = np.vstack((AN[0:50].T,AN[50:100].T,AN[100:150].T, \
                    AN[150:200].T,AN[200:250].T,AN[250:300].T, \
                    AN[300:350].T,AN[350:400].T,AN[400:450].T,\
                    AN[450:500].T,AN[500:550].T,AN[550:600].T)) #900X50
    AN_new = AN-AN_MeanMat #(3nN,m)=(900X50)
      
    # AN's mean
    A_MeanMat = np.tile(AN_MeanVec,(50,1))#50X75
    A2 = read_txt('A2.txt')
   
    #A0 = Tracking2D[0:50].astype(float)
    #A0 = Tracking2D[50:100].astype(float)
    #A0 = Tracking2D[100:150].astype(float)
    #A0 = Tracking2D[150:200].astype(float)
    #A0 = Tracking2D[200:250].astype(float)
    #A0 = Tracking2D[250:300].astype(float)
    #A0 = Tracking2D[300:350].astype(float)
    A0 = Tracking2D[350:400].astype(float)
    
   
    
    A0_new = A0-A_MeanMat#50X75
    A0_new = setzeros(A0_new,A2).T#75X50
   
    VN = mysvd(np.matmul(AN_new.T,AN_new)) 
    VV = np.matmul(VN,VN.T)
   
    V0 = mysvd(np.matmul(A0_new.T,A0_new)) 
    VV0 = np.matmul(V0,V0.T)
        
    F= np.matmul(V0.T,VN) 
    
    Bstar =  np.matmul(np.matmul(np.matmul(A0_new,V0),F),VN.T)#75X50
    BstarPM = Bstar.T + A_MeanMat  
    
    #A11 = read_txt('another_A1_df.txt')
    A11 = read_txt('A1.txt')
    A11 = setzeros(A11,A2)
    finalBstar = replaceZeros(A11,BstarPM)
    ref = read_txt('A1.txt')
    error_B= mse(ref,finalBstar)
    print('MSE of METHOD 2:',error_B)
    
    
     
    '''
    #######VISUALIZE#######
    Tracking2D=Tracking2D.ravel().reshape(Tracking2D.shape[0],25,3)
    #print('Tracking2D.shape:',Tracking2D.shape)

    for i in range(Tracking2D.shape[0]):
        fullfilenames = directory+'/'+files[i]
        img = cv2.imread(fullfilenames)
        Xs=[]
        Ys=[]
        for j in range(Tracking2D.shape[1]):
            x=round(float(Tracking2D[i][j][0]))
            y=round(float(Tracking2D[i][j][1]))
            Xs.append(x)
            Ys.append(y)
            #cv2.circle(img,(x,y),2,(0,0,0),thickness=-1)
        drawline(img,0,1,Xs,Ys,0)
        drawline(img,1,2,Xs,Ys,1)
        drawline(img,1,8,Xs,Ys,14)
        drawline(img,2,3,Xs,Ys,3)
        drawline(img,3,4,Xs,Ys,4)
        drawline(img,1,5,Xs,Ys,5)
        drawline(img,5,6,Xs,Ys,3)
        drawline(img,6,7,Xs,Ys,4)
        drawline(img,0,15,Xs,Ys,8)
        drawline(img,0,16,Xs,Ys,9)
        drawline(img,15,17,Xs,Ys,10)
        drawline(img,16,18,Xs,Ys,11)
        drawline(img,8,9,Xs,Ys,12)
        drawline(img,8,12,Xs,Ys,11)
        drawline(img,9,10,Xs,Ys,0)
        drawline(img,10,11,Xs,Ys,15)
        drawline(img,11,22,Xs,Ys,22)
        drawline(img,22,23,Xs,Ys,23)
        drawline(img,11,24,Xs,Ys,18)
        drawline(img,12,13,Xs,Ys,0)
        drawline(img,13,14,Xs,Ys,15)
        drawline(img,14,21,Xs,Ys,18)
        drawline(img,14,19,Xs,Ys,22)
        drawline(img,19,20,Xs,Ys,23)
        #cv.imshow("Image",img)
        #cv.waitKey(0)
        #print(i)
    
        for j in range(Tracking2D.shape[1]):
            x=round(float(Tracking2D[i][j][0]))
            y=round(float(Tracking2D[i][j][1]))
            Xs.append(x)
            Ys.append(y)
            cv2.circle(img,(x,y),2,(0,0,0),thickness=-1)
            
        out_img=save_directory +'/'+files[i]
        cv2.imwrite(out_img, img)
        video.write(img)
    video.release()
    
    cap = cv2.VideoCapture('003.avi')
     
    while(True):
        ret,frame = cap.read()#capture one frame
        if ret:
            cv2.imshow('frame',frame)
            cv2.waitKey(25)
            #cv.waitKey(0)
        else:
            break 
    cap.release()
    cv2.destroyAllWindows()
    '''
   
    

    
   


   
    
   
    
    
    
    
    
       
    
    
    
    
    
    
    
    