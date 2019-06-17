# -*- coding: utf-8 -*-
"""
interpolations on the A_1 
A_1 is created from a full matrix rather than a true deficient matrix
the difference between this code and **0508**on-A1 is that, 
this code constructs the fakeA1 basing on the full ori-A1
while **0508** wrongly use the full ori-A1 as A1
"""

import cv2
import os
import numpy as np
from scipy.linalg import lstsq

'''function that reads txt file'''
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
    #for i in range(100):
        #print(i)
        for j in range(len(Mat[i])):
            if Mat[i][j]==0:
                print([i,j],Mat[i][j])
                
def setzeros(A0,A1):
    for i in range(len(A1)):
        for j in range(len(A1[i])):
            if A1[i][j] == 0:
                #print([i,j],A1[i][j])
                #print([i,j])
                A0[i][j] = 0
    return A0
    

def findandsaveindex(Mat):
    list = []
    for i in range(len(Mat)):
        print(i)
        for j in range(len(Mat[i])):
            if Mat[i][j]==0:
                list.append([i,j]) 
                print([i,j],Mat[i][j])
    arr = np.array(list)
    arr = arr.astype(int)
    np.savetxt('indexes.txt',arr)
    
    
def txt2arr(filename):
    filename = 'xx.txt'
    file = open(filename,'r')
    list = file.readlines()
    lists = []
    for fields in list: 
        fields=fields.strip()# delete blank, including:'\n','\r','\t',' '
        fields=fields.strip('[]')#delete:[]
        fields=fields.split(',')# split by ','
        lists.append(fields)
    arr = np.array(lists)
    arr = arr.astype(int)
    print (arr)
    file.close()
  
'''
def save_cvs(txt_filename,cvs_filename):
    txt = np.loadtxt(txt_filename)  
    txtDF = pd.DataFrame(txt)     
    txtDF.to_csv(cvs_filename,index=False)
'''    

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
    files = os.listdir(directory)
    Tracking2D = []
    fullfilename = directory+'/'+files[0]
    
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
      
    A2 = read_txt('A2.txt')#50x75
    
    #A1 is the same with A0 since A1 is included in AN
    #A1 = Tracking2D[0:50].astype(float)#50x75
    A1 = Tracking2D[50:100].astype(float)
    #A1 = Tracking2D[100:150].astype(float)
    #A1 = Tracking2D[150:200].astype(float)
    #A1 = Tracking2D[200:250].astype(float)
    #A1 = Tracking2D[250:300].astype(float)
    #A1 = Tracking2D[300:350].astype(float)
    #A1 = Tracking2D[350:400].astype(float)
    
    A_MeanMat = np.tile(AN_MeanVec,(50,1))#50X75
    A1_new = A1-A_MeanMat#50X75
    A1_new = setzeros(A1_new,A2).T#75X50
    
    V1 = mysvd(np.matmul(A1_new.T,A1_new))#50x50
    VN = mysvd(np.matmul(AN_new.T,AN_new))#50x50 
    F= np.matmul(V1.T,VN) #the same with F= np.matmul(V0.T,VN) #50x50
    
    AN2 = Tracking2D[0:600].astype(float)#600x75      
    AN2_MeanVec = np.mean(AN2,0)
    AN2_MeanMat = np.tile(AN2_MeanVec,(AN2.shape[0],1))#600x75
    AN2_new = AN2-AN2_MeanMat#600x75  
    U1 = mysvd(np.matmul(A1_new,A1_new.T))#75x75
    UN = mysvd(np.matmul(AN2_new.T,AN2_new))#75x75   
    TMat = np.matmul(U1.T,UN)#75x75
    
    Qup = np.kron(np.eye(50),UN.T) # 3nm x 3nm=3750x3750
    Pup = np.matmul(np.matmul(TMat.T,U1.T),A1_new) # 3n x m=75x50
    Pup_vec = np.ravel(Pup,order='F') #3nmx1=3750x1
    Qdown = np.kron(VN.T,np.eye(75)) # 3nm x 3nm =3750x3750
    Pdown = np.matmul(np.matmul(A1_new,V1),F) # 3n x m=75x50
    Pdown_vec = np.ravel(Pdown,order='F') #3nmx1=3750x1
    Q = np.vstack((Qup,Qdown)) #7500x3750
    P = np.hstack((Pup_vec,Pdown_vec)) #7500x1
    X = lstsq(Q,P)[0] #3750x1
    Cstar=np.reshape(X,(50,75))#50x75
    CstarPM=Cstar+ A_MeanMat 
    #A13 = read_txt('another_A1_df.txt')
    A13 = read_txt('A1.txt')
    A13 = setzeros(A13,A2)
    finalCstar= replaceZeros(A13,CstarPM)
    ref = read_txt('A1.txt')
    error_C= mse(ref,finalCstar)
    print('MSE of METHOD 3:',error_C)
    
    print('Finished!')
    
    
    
    
   
    
    
    
    
    
       
    
    
    
    
    
    
    
    