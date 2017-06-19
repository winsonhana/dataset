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
from conv3D import Conv3D_Tranpose1, MaxPool3D
import matplotlib.pyplot as plt

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
        #poolStride = (1,1,1)
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
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=1, output_shape=img, kernel_size=(3,3,3), stride=(2,2,2), padding='SAME'))
        #seq.add(Softmax())
        seq.add(Sigmoid())
    return seq
        

if __name__ == '__main__':

    learning_rate = 0.001
    batchsize = 2
    split = 45 # Train Valid Split
    
    max_epoch = 300
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)


    seq = model3D()
    dataset = WMHdataset('./WMH')
    assert dataset.AbleToRetrieveData(), 'not able to locate the directory of dataset'
    dataset.InitDataset(split=1.0)         # Take everything 100%

    #X_ph = tf.placeholder('float32', [None, 83, 256, 256, 1])
    #y_ph = tf.placeholder('float32', [None, 83, 256, 256, 1])
    X_ph = tf.placeholder('float32', [None, None, None, None, 1])
    y_ph = tf.placeholder('float32', [None, None, None, None, 1])
    
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)

    #### COST FUNCTION
    #train_cost_sb = tf.reduce_mean((y_ph - y_train_sb)**2)
    train_cost_sb = entropy(y_ph, y_train_sb)

    #test_cost_sb = tf.reduce_mean((y_ph - y_test_sb)**2)
    test_cost_sb = entropy(y_ph, y_test_sb)
    test_accu_sb = accuracy(y_ph, y_test_sb)
    #test_accu_sb = smooth_iou(y_ph, y_test_sb)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_sb)
    
    # model Saver
    saver = tf.train.Saver()
    
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("INITIALIZE SESSION")
        
        dataX, dataY = dataset.NextBatch3D(60) # Take everything
        #######
        # Just to train 0 & 1, ignore 2=Other Pathology. Assign 2-->0
        dataY[dataY ==2] = 0
        #X_train = dataX[:split]
        X_test = dataX[split:]
        #y_train = dataY[:split]
        y_test = dataY[split:]
        dataX = []
        dataY = []
        print("PREDICTING")
        #######
        predictIndex = 0
        saver.restore(sess, "trained_model.ckpt")
        feed_dict = {X_ph:X_test[predictIndex].reshape((1,)+X_test[0].shape),
                     y_ph:y_test[predictIndex].reshape((1,)+X_test[0].shape)}
        valid_cost, valid_accu = sess.run([test_cost_sb, test_accu_sb] , feed_dict=feed_dict)
        mask_output = sess.run(y_test_sb, feed_dict=feed_dict)
        mask_output = (mask_output > 0.5).astype(int)
        mask_output = mask_output * 255.0
        ####### Plotting
        slice = 25
        cmap_ = 'CMRmap'
        plt.figure(figsize=(7,7))
        plt.subplot(2,2,1)
        plt.imshow(X_test[predictIndex,slice,:,:,0], cmap_)
        plt.title('Flair Image')
        plt.subplot(2,2,2)
        plt.imshow(mask_output[predictIndex,slice,:,:,0], cmap_)
        plt.title('Predicted Mask, accuracy: %s' % valid_accu)
        plt.subplot(2,2,3)
        plt.imshow(y_test[predictIndex,slice,:,:,0], cmap_)
        plt.title('Actual Mask')
        plt.tight_layout()
        
        

