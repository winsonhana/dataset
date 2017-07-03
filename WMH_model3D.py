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
from tensorgraph.node import StartNode, HiddenNode, EndNode
from tensorgraph.graph import Graph
from tensorgraph.layers.merge import Concat, Mean, Sum
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
from math import ceil
from WMH_loadData import WMHdataset # 3D MRI Scanned Dataset
from conv3D import Conv3D_Tranpose1, MaxPool3D, SoftMaxMultiDim, Residual3D, InceptionResnet_3D



####

def updateConvLayerSize(dataDimension,stride):
    assert len(dataDimension) == len(stride), "TensorRank of dataset is not the same as stride's rank."
    output_ = tuple()
    for i in range(len(stride)):
        output_ += (int(ceil(dataDimension[i]/float(stride[i]))),)
    return output_



#def model3D(img=(None, None, None)):
def model3D(img=(83, 256, 256)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        seq.add(Conv3D(input_channels=1, num_filters=10, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)
        #print("layer1: "+str(layerSize1))
        seq.add(RELU())
        seq.add(Conv3D(input_channels=10, num_filters=20, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(TFBatchNormalization(name='b2'))
        #layerSize2 = updateConvLayerSize(layerSize1,convStride)
        #print("layer1: "+str(layerSize2))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=20, num_filters=10, output_shape=layerSize1, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=10, num_filters=3, output_shape=img, kernel_size=(3,3,3), stride=(2,2,2), padding='SAME'))
        ##        
        seq.add(RELU())        
        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        ##
        seq.add(Softmax())
        #seq.add(Sigmoid())
    return seq
        

def model3D_2(img=(83, 256, 256)):
    with tf.name_scope('WMH_2Chan_Input'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=convStride, padding='SAME'))        
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
        ##        
        #seq.add(RELU())        
        #seq.add(Conv3D(input_channels=2, num_filters=2, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
#        seq.add(RELU())        
#        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
#        seq.add(RELU())        
#        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
#                
        ##
        seq.add(Softmax())
        #seq.add(Sigmoid())
    return seq


def model3D_Residual(img=(83, 256, 256)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        kernelSize = (3,3,3)
        
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=convStride, padding='SAME'))        
        #seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)    
        seq.add(RELU())
        
        seq.add(Residual3D(input=8,num_blocks=3,kernel=kernelSize))
        
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize2 = updateConvLayerSize(layerSize1,poolStride)
        seq.add(RELU())
        
        seq.add(Residual3D(input=16,num_blocks=3,kernel=kernelSize))
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=16, output_shape=layerSize2, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        seq.add(RELU())
        
        seq.add(Residual3D(input=16,num_blocks=3,kernel=kernelSize))        
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(RELU())
        
        # num_filter=3 --> Background, WhiteMatter, Others
        seq.add(Conv3D(input_channels=8, num_filters=8, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=3, output_shape=img, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        ##        
        seq.add(RELU())        
        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(1,1,1), stride=convStride, padding='SAME'))        
        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(1,1,1), stride=convStride, padding='SAME'))        
        ##  
        seq.add(Softmax())
    return seq
    
def model3D_ResidualDeeper(img=(83, 256, 256)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        kernelSize = (3,3,3)
        
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=convStride, padding='SAME'))        
        #seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)    
        seq.add(RELU())
        
        seq.add(Residual3D(input=8,num_blocks=3,kernel=kernelSize))
        
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize2 = updateConvLayerSize(layerSize1,poolStride)
        seq.add(RELU())
        
        seq.add(Residual3D(input=16,num_blocks=3,kernel=kernelSize))
        
        seq.add(Conv3D(input_channels=16, num_filters=32, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize3 = updateConvLayerSize(layerSize2,poolStride)
        seq.add(RELU())
        
        seq.add(Residual3D(input=32,num_blocks=3,kernel=kernelSize))
        
        seq.add(Conv3D_Tranpose1(input_channels=32, num_filters=16, output_shape=layerSize2, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(RELU())
        
        seq.add(Residual3D(input=16,num_blocks=3,kernel=kernelSize))        
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(RELU())
        
        # num_filter=3 --> Background, WhiteMatter, Others
        seq.add(Conv3D(input_channels=8, num_filters=8, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=3, output_shape=img, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        ##        
        seq.add(RELU())        
        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(1,1,1), stride=convStride, padding='SAME'))        
        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(1,1,1), stride=convStride, padding='SAME'))        
        ##  
        seq.add(Softmax())
    return seq    

def model_Inception_Resnet(img=(83, 256, 256)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        kernelSize = (3,3,3)
        
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=convStride, padding='SAME'))        
        #seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)    
        seq.add(RELU())
        
        seq.add(InceptionResnet_3D(8, type='v2_out8'))
        
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize2 = updateConvLayerSize(layerSize1,poolStride)
        seq.add(RELU())
        
        seq.add(InceptionResnet_3D(16, type='v1_out16'))
        seq.add(Conv3D(input_channels=16, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        seq.add(RELU())
        
        seq.add(Conv3D(input_channels=16, num_filters=32, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b3'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize3 = updateConvLayerSize(layerSize2,poolStride)
        seq.add(RELU())        
        
        seq.add(InceptionResnet_3D(32, type='v1_out16'))    
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=16, output_shape=layerSize2, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(RELU())
        
        seq.add(InceptionResnet_3D(16, type='v1_out16'))  
        seq.add(Conv3D(input_channels=16, num_filters=16, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        seq.add(RELU())
        
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        seq.add(RELU())  
        
        seq.add(InceptionResnet_3D(8, type='v2_out8'))         
        seq.add(Conv3D(input_channels=8, num_filters=8, kernel_size=kernelSize, stride=convStride, padding='SAME'))
        seq.add(RELU())
        
        # num_filter=3 --> Background, WhiteMatter, Others
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=3, output_shape=img, kernel_size=kernelSize, stride=poolStride, padding='SAME'))
        ##        
        seq.add(RELU())        
        seq.add(Conv3D(input_channels=3, num_filters=3, kernel_size=(1,1,1), stride=convStride, padding='SAME'))        
        ##  
        seq.add(Softmax())
    return seq
    
    
#def VNet(img=(83, 256, 256)):
#    with tf.name_scope('WMH'):
#        seq = tg.Sequential()
#        convStride = (1,1,1)
#        poolStride = (2,2,2)
#        kSize3 = (3,3,3)
#        kSize5 = (5,5,5)
#        seq.add(Conv3D(input_channels=1, num_filters=16, kernel_size=kSize5, stride=convStride, padding='SAME'))        
#        
#        
#        
#        graph =         
#    return graph
#        
#        x_dim = 50
#        component_dim = 100
#        batchsize = 32
#        learning_rate = 0.01
#        x_ph = tf.placeholder('float32', [None, x_dim])
#        start = StartNode(input_vars=[x_ph])
#        h1 = HiddenNode(prev=[start], layers=[Linear(x_dim, component_dim), Softmax()])
#        e1 = EndNode(prev=[h1], input_merge_mode=Sum())
#        #e3 = EndNode(prev=[h1, h2, h3], input_merge_mode=Sum())
#        
#        graph = Graph(start=[start], end=[e1, e2, e3])
#        o1, o2, o3 = graph.train_fprop()
#        o1_mse = tf.reduce_mean((y1_ph - o1)**2)
#        o2_mse = tf.reduce_mean((y2_ph - o2)**2)
#        
#        
#        