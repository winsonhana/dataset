# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 16:37:23 2017

@author: winsoncws

Attempt 3D ConvNet
"""
from __future__ import division, print_function, absolute_import

from tensorgraph.layers import Conv3D, Conv2D, RELU, MaxPooling, LRN, Tanh, Dropout, \
                               Softmax, Flatten, Linear, TFBatchNormalization, Sigmoid
from tensorgraph.utils import same
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
from math import ceil
from WMH_loadData import WMHdataset # 3D MRI Scanned Dataset
from conv3D import Conv3D_Tranpose1, MaxPool3D, SoftMaxMultiDim, Residual3D



####

def updateConvLayerSize(dataDimension,stride):
    assert len(dataDimension) == len(stride), "TensorRank of dataset is not the same as stride's rank."
    output_ = tuple()
    for i in range(len(stride)):
        output_ += (int(ceil(dataDimension[i]/float(stride[i]))),)
    return output_




def model3D(img=(83, 256, 256)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)
        #print("layer1: "+str(layerSize1))
        seq.add(RELU())
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(TFBatchNormalization(name='b2'))
        #layerSize2 = updateConvLayerSize(layerSize1,convStride)
        #print("layer1: "+str(layerSize2))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=2, output_shape=img, kernel_size=(3,3,3), stride=(2,2,2), padding='SAME'))
        seq.add(Softmax())
        #seq.add(Sigmoid())
    return seq
        

def model3D_2(img=(83, 256, 256)):
    with tf.name_scope('WMH_2Chan_Input'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        kernelSize = (5,5,5)
        seq.add(Conv3D(input_channels=2, num_filters=8, kernel_size=kernelSize, stride=convStride, padding='SAME'))        
        #seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)
        #print("layer1: "+str(layerSize1))
        seq.add(RELU())
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        #layerSize2 = updateConvLayerSize(layerSize1,convStride)
        #print("layer1: "+str(layerSize2))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        seq.add(RELU())
        # num_filter=3 --> Background, WhiteMatter, Others
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=3, output_shape=img, kernel_size=kernelSize, stride=(2,2,2), padding='SAME'))       
        seq.add(Softmax())
        #seq.add(Sigmoid())
    return seq


def model3D_Residual(img=(83, 256, 256)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        kernelSize = (3,3,3)
        
        seq.add(Conv3D(input_channels=2, num_filters=8, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        #seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)    
        seq.add(RELU())
        
        #seq.add(Residual3D(input=8,num_blocks=3,kernel=kernelSize))
        
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize2 = updateConvLayerSize(layerSize1,poolStride)
        seq.add(RELU())
        
        #seq.add(Residual3D(input=20,num_blocks=3,kernel=kernelSize))
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=16, output_shape=layerSize2, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        seq.add(RELU())
        
        #seq.add(Residual3D(input=16,num_blocks=3,kernel=kernelSize))        
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(RELU())
        
        # num_filter=3 --> Background, WhiteMatter, Others
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=3, output_shape=img, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(Softmax())
    return seq