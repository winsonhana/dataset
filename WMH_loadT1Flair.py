
"""
Created on Tue Jun 13 10:05:31 2017

@author: winsoncws
"""

import os
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.interpolation import rotate





####### Iterating 3D Scanning Data ########

class WMHdataset():
    
    institute = ['Utrecht','Singapore','GE3T']
    #institute = ['Utrecht']
    trainIndex = 0
    validIndex = 0
    listFolder = []
    listTrain = []
    listValid = []
    
    def __init__(self, filepath):
        assert os.path.exists(filepath), "no such path directory"
        self.filepath = filepath
        
    def SetPath(self, filepath):
        assert os.path.exists(filepath), "no such path directory"
        self.filepath = filepath
      
    def AbleToRetrieveData(self):
        if os.path.exists(os.path.join(self.filepath,self.institute[0],'6')):
            return True
        else: 
            return False
    
    def InitDataset(self,splitRatio=1.0, shuffle=False):
        np.random.seed(0) # set seed for reproducibility
        assert self.AbleToRetrieveData(), "not able to retrieve data from path."
        for i in self.institute:
            path_ = os.path.join(self.filepath,i)
            listDir = os.listdir(path_)
            listDir = [os.path.join(i,f) for f in listDir if os.path.exists(os.path.join(self.filepath,i,f,'pre'))]       
            self.listFolder = self.listFolder + listDir
        length_ = len(self.listFolder)
        if shuffle == False:
            sampleIndex = range(length_)[:int(np.round(length_*splitRatio))]
        else:
            sampleIndex = np.random.choice(range(length_),int(np.round(length_*splitRatio)),False)
        self.listTrain = [self.listFolder[i] for i in sampleIndex]
        self.listValid = [self.listFolder[i] for i in range(length_) if i not in sampleIndex]


    def nextBatch(self,batchSize,dataset='train'):
        if dataset == 'train':
            dataFolder = self.listTrain
            batchIndex = self.trainIndex
        else:
            dataFolder = self.listValid
            batchIndex = self.validIndex
        length = len(dataFolder)
        batchEnd = batchIndex + batchSize
        start_ = np.mod(batchIndex,length)
        end_ = np.mod(batchEnd,length)
        if batchSize == length:
            batchPath_ = dataFolder
        elif end_ < (start_):
            batchPath_ = dataFolder[start_:]+dataFolder[:(end_)]
        else:
            batchPath_ = dataFolder[start_:end_]
        if dataset == 'train':
            self.trainIndex = batchEnd
        else:
            self.validIndex = batchEnd
        return batchPath_
    
    def checkAllDim(self):
        fullPathsFlair_ = [os.path.join(self.filepath,i,'pre','FLAIR.nii.gz') for i in self.listFolder]
        print('dimension for all FLAIR Images')        
        for i in fullPathsFlair_:
            print(sitk.GetArrayFromImage(sitk.ReadImage(i)).shape)

        
    def padding(self,x,dim=(83,256,256)):
        dataShape = x.shape
        d1 = int(np.ceil((dim[0]-dataShape[0])/2.0))
        d2 = int(np.floor((dim[0]-dataShape[0])/2.0))
        w1 = int(np.ceil((dim[1]-dataShape[1])/2.0))
        w2 = int(np.floor((dim[1]-dataShape[1])/2.0))
        h1 = int(np.ceil((dim[2]-dataShape[2])/2.0))
        h2 = int(np.floor((dim[2]-dataShape[2])/2.0))
        return np.pad(x,[[d1,d2],[w1,w2],[h1,h2]],'constant')
    
    def RotateTopAxis(self,data,angle):
        return rotate(data,angle, axes=(1,2) , reshape=False )
    
    
    def NextBatch3D(self,batchSize,dataset='train',subfolder='pre'):
        batchPath_ = self.nextBatch(batchSize,dataset)
        fullPathsFlair_ = [os.path.join(self.filepath,i,subfolder,'FLAIR.nii.gz') for i in batchPath_]
        #fullPathsT1_ = [os.path.join(self.filepath,i,subfolder,'T1.nii.gz') for i in batchPath_]
        fullPathsWMH_ = [os.path.join(self.filepath,i,'wmh.nii.gz') for i in batchPath_]
        
        print('fetching '+dataset+' rawdata from drive')
        # maxValue = 3180.0
        maxValue = 1.0
        #dataT1_ = [self.padding(sitk.GetArrayFromImage(sitk.ReadImage(i)))/maxValue for i in fullPathsT1_]
        dataFl_ = [self.padding(sitk.GetArrayFromImage(sitk.ReadImage(i)))/maxValue for i in fullPathsFlair_]  
        #dataX_ = np.stack((dataT1_,dataFl_),axis=-1) # merge into 2 channels
        if dataset == 'train':
            dataX_ = []  
            for i in dataFl_:
                dataX_.append(i)
                dataX_.append(self.RotateTopAxis(i,10))
                dataX_.append(self.RotateTopAxis(i,-10))
            dataX_ = np.array([i.reshape(i.shape+(1,)) for i in dataX_])
        else:
            dataX_ = np.array([i.reshape(i.shape+(1,)) for i in dataFl_])
        dataLabel_ = [self.padding(sitk.GetArrayFromImage(sitk.ReadImage(i))) for i in fullPathsWMH_]
        dataY_ = []
        if dataset == 'train':
            dataY_ = []  
            for i in dataLabel_:
                dataY_.append(i)
                dataY_.append(self.RotateTopAxis(i,10))
                dataY_.append(self.RotateTopAxis(i,-10))
            dataY_ = np.array([i.reshape(i.shape+(1,)) for i in dataY_])
        else:
            dataY_ = np.array([i.reshape(i.shape+(1,)) for i in dataLabel_])
        print('retrieved rawdata from drive')
        return dataX_, dataY_

        
    def showImages(self,scan=13,slice=13, cmap_='CMRmap'):
        path_ = self.listFolder[scan]
        path_ = os.path.join(self.filepath,path_) 
        imgT1 = sitk.GetArrayFromImage(sitk.ReadImage(path_+'/pre/T1.nii.gz'))
        imgFlair = sitk.GetArrayFromImage(sitk.ReadImage(path_+'/pre/FLAIR.nii.gz'))
        imgFlairOri = sitk.GetArrayFromImage(sitk.ReadImage(path_+'/orig/FLAIR.nii.gz'))
        imgWMH = sitk.GetArrayFromImage(sitk.ReadImage(path_+'/wmh.nii.gz'))
        plt.figure(figsize=(7,7))
        plt.subplot(2,2,1)
        plt.imshow(imgT1[slice,:,:], cmap_)
        plt.title('T1 Image')
        plt.subplot(2,2,2)
        plt.imshow(imgFlair[slice,:,:], cmap_)
        plt.title('Flair Bias Corrected')
        plt.subplot(2,2,3)
        plt.imshow(imgWMH[slice,:,:], cmap_)
        plt.title('WMH Label')
        plt.subplot(2,2,4)
        plt.imshow(imgFlairOri[slice,:,:], cmap_)
        plt.title('Flair Original')
        plt.tight_layout()
        
    
        
###     
#DLpath2 = '/Users/winsoncws/Hana/WMH/' 
#D = WMHdataset(DLpath2)
#D.InitDataset()
#
#dataX , dataY = D.NextBatch3D(2)    
#maxX = 0
#for i in dataY:
#    if maxX < i.max():
#        maxX = i.max()
## return 3180  
#minX = 0
#for i in dataY:
#    if minX > i.max():
#        minX = i.max()
# return 0

# Utrecht   (48, 240, 240, 1)
# Singapore (48, 256, 232, 1)
# Amsterdam (83, 256, 132, 1)
# MAX (83,256,240,1)


#DLpath = '/Users/winsoncws/Downloads'
########################################
######### USING SITK PACKAGE ###########
########################################
#
#T1sitk = sitk.ReadImage('./dataset/Utrecht/0/pre/T1.nii.gz')
#T1sitk = sitk.GetArrayFromImage(T1sitk)
#plt.imshow(T1sitk[20,:,:])


#def showScanImage(path,index, cmap_='CMRmap'):
#    imgT1 = sitk.GetArrayFromImage(sitk.ReadImage(path+'/pre/T1.nii.gz'))
#    imgFlair = sitk.GetArrayFromImage(sitk.ReadImage(path+'/pre/FLAIR.nii.gz'))
#    imgFlairOri = sitk.GetArrayFromImage(sitk.ReadImage(path+'/orig/FLAIR.nii.gz'))
#    imgWMH = sitk.GetArrayFromImage(sitk.ReadImage(path+'/wmh.nii.gz'))
#    plt.figure(figsize=(7,7))
#    plt.subplot(2,2,1)
#    plt.imshow(imgT1[index,:,:], cmap_)
#    plt.title('T1 Image')
#    plt.subplot(2,2,2)
#    plt.imshow(imgFlair[index,:,:], cmap_)
#    plt.title('Flair Bias Corrected')
#    plt.subplot(2,2,3)
#    plt.imshow(imgWMH[index,:,:], cmap_)
#    plt.title('WMH Label')
#    plt.subplot(2,2,4)
#    plt.imshow(imgFlairOri[index,:,:], cmap_)
#    plt.title('Flair Original')
#    plt.tight_layout()


### SHOW ONE OF THE SCAN IMAGE ##########
#showScanImage(DLpath+'/0',20,'CMRmap')
