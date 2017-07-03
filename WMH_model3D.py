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
from tensorgraph.layers.merge import Concat, Mean, Sum, NoChange
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
from math import ceil
from WMH_loadData import WMHdataset # 3D MRI Scanned Dataset
from conv3D import Conv3D_Tranpose1, MaxPool3D, SoftMaxMultiDim, Residual3D, \
InceptionResnet_3D, ResidualBlock3D


####

def updateConvLayerSize(dataDimension,stride):
    assert len(dataDimension) == len(stride), "TensorRank of dataset is not the same as stride's rank."
    output_ = tuple()
    for i in range(len(stride)):
        output_ += (int(ceil(dataDimension[i]/float(stride[i]))),)
    return output_



#def model3D(img=(None, None, None)):
def model3D(img=(84, 256, 256)):
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
        

def model3D_2(img=(84, 256, 256)):
    with tf.name_scope('WMH_2Chan_Input'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(5,5,5), stride=convStride, padding='SAME'))        
        seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)
        seq.add(RELU())
        
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(TFBatchNormalization(name='b2'))
        
        ## Extra MaxPool
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        #layerSize2 = updateConvLayerSize(layerSize1,convStride)
        seq.add(RELU())
        ## Extra Conv
        seq.add(Conv3D(input_channels=16, num_filters=16, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(TFBatchNormalization(name='b3'))
        
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=(3,3,3), stride=poolStride, padding='SAME'))
        seq.add(TFBatchNormalization(name='b4'))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=2, output_shape=img, kernel_size=(3,3,3), stride=poolStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b5'))
        #seq.add(RELU())
        
        #seq.add(Conv3D(input_channels=2, num_filters=2, kernel_size=(1,1,1), stride=convStride, padding='SAME'))
        ##
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


def model3D_Residual(img=(84, 256, 256)):
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
    
def model3D_ResidualDeeper(img=(84, 256, 256)):
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
        #layerSize3 = updateConvLayerSize(layerSize2,poolStride)
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

def model_Inception_Resnet(img=(84, 256, 256)):
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
        #layerSize3 = updateConvLayerSize(layerSize2,poolStride)
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
    
class Testing():
    def __init__(self,input):
        self.int = input
    def _train_fprop(self, state_below):
        print("testing "+str(self.int))
        return state_below
    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)

def Residual_UNET(input, img=(84, 256, 256)):
    with tf.name_scope('WMH'):
        convStride = (1,1,1)
        poolStride = (2,2,2)
        kSize3 = (3,3,3)
        #kSize5 = (5,5,5)
        
        #x_dim = 50
        #component_dim = 100
        #batchsize = 32
        #learning_rate = 0.01
        #x_ph = tf.placeholder('float32', [None, x_dim])
        #start = StartNode(input_vars=[x_ph])
        #h1 = HiddenNode(prev=[start], layers=[Linear(x_dim, component_dim), Softmax()])
        #e1 = EndNode(prev=[h1], input_merge_mode=Sum())
        #e3 = EndNode(prev=[h1, h2, h3], input_merge_mode=Sum())
        
        start = StartNode(input_vars=[input])
        
        Layer01 = [Conv3D(input_channels=1, num_filters=8, kernel_size=kSize3, stride=convStride, padding='SAME') ]
        #Layer01.append(RELU())        
        
        LayerPool = MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME')
        
        Layer02 = [LayerPool]
        #Layer02.append(Testing(1))
        Layer02.append(Conv3D(input_channels=8, num_filters=16, kernel_size=kSize3, stride=convStride, padding='SAME') )
        #Layer02.append(RELU())  
        #Layer02.append(Testing(2))        
        Layer02.append(ResidualBlock3D(16,'L02'))
        #Layer02.append(Testing(3))
        layerSize1 = updateConvLayerSize(img,poolStride)  
        
        Layer03 = [LayerPool]
        Layer03.append(Conv3D(input_channels=16, num_filters=32, kernel_size=kSize3, stride=convStride, padding='SAME') )
        #Layer03.append(RELU())  
        Layer03.append(ResidualBlock3D(32,'L03'))
        #layerSize2 = updateConvLayerSize(layerSize1,poolStride)
        Layer03.append(Conv3D_Tranpose1(input_channels=32, num_filters=16, output_shape=layerSize1, kernel_size=kSize3, stride=poolStride, padding='SAME') )
        #Layer03.append(RELU())  
        
        #Layer04 = [Conv3D(input_channels=32, num_filters=64, kernel_size=kSize5, stride=convStride, padding='SAME')]
        #Layer04.append(ResidualBlock3D(64,'L03'))
        
        conv8 = HiddenNode(prev=[start], layers=Layer01)
        
        resBlock16 = HiddenNode(prev=[conv8], layers=Layer02)

        resBlock32_16 = HiddenNode(prev=[resBlock16], layers=Layer03)
        residualLong16 = HiddenNode(prev=[resBlock32_16,resBlock16], input_merge_mode=Sum())
        

        Layer04 = [ResidualBlock3D(16,'L04')]
        Layer04.append(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=img, kernel_size=kSize3, stride=poolStride, padding='SAME') )
        Layer04.append(RELU())
        
        resBlock16_8 = HiddenNode(prev=[residualLong16], layers=Layer04)
        residualLong8 = HiddenNode(prev=[resBlock16_8,conv8], input_merge_mode=Sum())
        
        Layer05 = [ResidualBlock3D(8,'L05')]
        Layer05.append(Conv3D(input_channels=8, num_filters=2, kernel_size=kSize3, stride=convStride, padding='SAME') )
        Layer05.append(Softmax())        
        
        resBlock8_2 = HiddenNode(prev=[residualLong8], layers=Layer05)
        
        endNode = EndNode(prev=[resBlock8_2], input_merge_mode=NoChange())
        
        graph = Graph(start=[start], end=[endNode])
        #o1, o2, o3 = graph.train_fprop()
        #o1_mse = tf.reduce_mean((y1_ph - o1)**2)
        #o2_mse = tf.reduce_mean((y2_ph - o2)**2)
        
    return graph
        