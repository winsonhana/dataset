# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:24:48 2017

@author: winsoncws
"""
import sys
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
from WMH_loadData import WMHdataset # 3D MRI Scanned Dataset
from conv3D import Conv3D_Tranpose1, MaxPool3D
#import matplotlib.pyplot as plt
import WMH_model3D # all model
from scipy.misc import imsave
import numpy as np

if __name__ == '__main__':

    learning_rate = 0.001
    batchsize = 6
    split = 48 # Train Valid Split
    
    max_epoch = 100
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)


    seq = WMH_model3D.model3D()
    dataset = WMHdataset('./WMH')
    assert dataset.AbleToRetrieveData(), 'not able to locate the directory of dataset'
    dataset.InitDataset(split=1.0)         # Take everything 100%

    X_ph = tf.placeholder('float32', [None, 83, 256, 256, 1])
    y_ph = tf.placeholder('float32', [None, 83, 256, 256, 1])
    #X_ph = tf.placeholder('float32', [None, None, None, None, 1])
    #y_ph = tf.placeholder('float32', [None, None, None, None, 1])
    
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)

    #### COST FUNCTION
    #train_cost_sb = tf.reduce_mean((y_ph - y_train_sb)**2)
    #train_cost_sb = entropy(y_ph, y_train_sb)
    train_cost_sb = 1 - smooth_iou(y_ph, y_train_sb)

    #test_cost_sb = tf.reduce_mean((y_ph - y_test_sb)**2)
    test_cost_sb = entropy(y_ph, y_test_sb)
    #test_accu_sb = accuracy(y_ph, y_test_sb)
    test_accu_sb = iou(y_ph, y_test_sb, threshold=0.2)

    print('DONE')    

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_sb)
    
    # model Saver
    saver = tf.train.Saver()
    
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("INITIALIZE SESSION")
        
        #sys.exit() 
        #tf.reduce_max()
        
        dataX, dataY = dataset.NextBatch3D(60) # Take everything
        #######
        # Just to train 0 & 1, ignore 2=Other Pathology. Assign 2-->0
        dataY[dataY ==2] = 0
        #######
        X_train = dataX[:split]
        X_test = dataX[split:]
        y_train = dataY[:split]
        y_test = dataY[split:]
        dataX = [] # clearing memory
        dataY = [] # clearing memory

        
        #save_path = saver.save(sess, "trained_model.ckpt")    
        #print("Model saved in file: %s" % save_path)
        saver.restore(sess, "trained_model.ckpt")
        
        # PREDICTION
        predictIndex = 6
        feed_dict = {X_ph:X_test[predictIndex].reshape((1,)+X_test[0].shape)}
#                     y_ph:y_test[predictIndex].reshape((1,)+X_test[0].shape)}
#       valid_cost, valid_accu = sess.run([test_cost_sb, test_accu_sb] , feed_dict=feed_dict)
        mask_output = sess.run(y_test_sb, feed_dict=feed_dict)

        print('mask_output type')        
        print(type(mask_output))
        #mask_output = (mask_output > 0.5).astype(int)
        #mask_output = mask_output * 255.0
        print(mask_output.shape)        
        
        np.save('X_test.npy',X_test[predictIndex])
        np.save('y_test.npy',y_test[predictIndex])
        np.save('mask_output.npy',mask_output[0])
        

