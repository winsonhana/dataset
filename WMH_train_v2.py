# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:24:48 2017

@author: winsoncws
"""
import sys
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou
#from WMH_loadData import WMHdataset # 3D MRI Scanned Dataset
from WMH_loadT1Flair import WMHdataset # 3D MRI Scanned Dataset
from conv3D import Conv3D_Tranpose1, MaxPool3D
#import matplotlib.pyplot as plt
import WMH_model3D # all model
from scipy.misc import imsave
import numpy as np

if __name__ == '__main__':
    
    #print(sys.argv[0]) # input from terminal
    #print(sys.argv[1]) # input from terminal
    #print(sys.argv[2]) # input from terminal
    
    learning_rate = 0.001
    
    max_epoch = 100
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)


    seq = WMH_model3D.model3D_Residual()
    dataset = WMHdataset('./WMH')
    assert dataset.AbleToRetrieveData(), 'not able to locate the directory of dataset'
    dataset.InitDataset(splitRatio=1.0, shuffle=True)         # Take everything 100%
    X_ph = tf.placeholder('float32', [None, 83, 256, 256, 1])  #float32
    y_ph = tf.placeholder('uint8', [None, 83, 256, 256, 1])
    #X_ph = tf.placeholder('float32', [None, None, None, None, 1])
    #y_ph = tf.placeholder('float32', [None, None, None, None, 1])
    #y_ph_cat = y_ph[:,:,:,:,0]   # # Works for Softmax filter2
    
    y_ph_cat = tf.one_hot(y_ph,3) # --> unstack into 3 categorical Tensor [?, 83, 256, 256, 1, 3]
    y_ph_cat = tf.reduce_max(y_ph_cat, 4)   # --> collapse the extra 4th redundant dimension
    
    
    # works for Label01 filter2
    #y_train_sb = (seq.train_fprop(X_ph))[:,:,:,:,1]   # works! but change the reshape
    #y_test_sb = (seq.test_fprop(X_ph))[:,:,:,:,1]       # works! maybe new variable
    print('TRAINING')
    # for one hot
    y_train_sb = (seq.train_fprop(X_ph))  
    y_test_sb = (seq.test_fprop(X_ph))   
    print('TRAINED')
    train_cost_background = (1 - smooth_iou(y_ph_cat[:,:,:,:,0] , y_train_sb[:,:,:,:,0]) )*0.001
    train_cost_label = (1 - smooth_iou(y_ph_cat[:,:,:,:,1] , y_train_sb[:,:,:,:,1]) )*0.5
    train_cost_others = (1 - smooth_iou(y_ph_cat[:,:,:,:,2] , y_train_sb[:,:,:,:,2]) )*0.499
    train_cost_sb = tf.reduce_sum([train_cost_background,train_cost_label,train_cost_others])
    valid_cost_background = (1 - smooth_iou(y_ph_cat[:,:,:,:,0] , y_test_sb[:,:,:,:,0]) )*0.001
    valid_cost_label = (1 - smooth_iou(y_ph_cat[:,:,:,:,1] , y_test_sb[:,:,:,:,1]) )*0.5
    valid_cost_others = (1 - smooth_iou(y_ph_cat[:,:,:,:,2] , y_test_sb[:,:,:,:,2]) )*0.499
    test_cost_sb = tf.reduce_sum([valid_cost_background,valid_cost_label,valid_cost_others])  

    #### COST FUNCTION
    #train_cost_sb = tf.reduce_mean((y_ph - y_train_sb)**2)
    #train_cost_sb = entropy(y_ph, y_train_sb)

    #train_cost_sb = 1 - smooth_iou(y_ph_cat, y_train_sb)          # Works for Softmax filter2

    #test_cost_sb = tf.reduce_mean((y_ph - y_test_sb)**2)
    #test_cost_sb = entropy(y_ph_cat, y_test_sb)                   # Works for Softmax filter2
    #test_accu_sb = accuracy(y_ph, y_test_sb)
    test_accu_sb = iou(y_ph_cat, y_test_sb, threshold=0.2)         # Works for Softmax filter2

    print('DONE')    
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_sb)
    
    # model Saver
    saver = tf.train.Saver()
    
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("INITIALIZE SESSION")

        #batchsize = 6
        #split = 48 # Train Valid Split
        #dataX, dataY = dataset.NextBatch3D(60) # Take everything
        #######
        # Just to train 0 & 1, ignore 2=Other Pathology. Assign 2-->0
        #dataY[dataY ==2] = 0
        #######
        #X_train = dataX[:split]
        #X_train = X_train[:,:,:,:,1]
        #X_train = X_train.reshape(X_train.shape+(1,))
        #X_test = dataX[split:]
        #X_test = X_test[:,:,:,:,1]
        #X_test = X_test.reshape(X_test.shape+(1,))
        #y_train = dataY[:split]
        #y_test = dataY[split:]
        
        dataset.InitDataset(splitRatio=1.0, shuffle=True)  # Take everything 100%
        
        batchsize = 5  # size=3
        #######
        # Just to train 0 & 1, ignore 2=Other Pathology. Assign 2-->0
        # dataY[dataY ==2] = 0
        #######
        #X_train, y_train = dataset.NextBatch3D(len(dataset.listTrain),dataset='train')
        #X_test, y_test = dataset.NextBatch3D(len(dataset.listValid),dataset='validation')
        X_train, y_train = dataset.NextBatch3D(45,dataset='train')
        X_test, y_test = dataset.NextBatch3D(12,dataset='validation')
        
        iter_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
        iter_test = tg.SequentialIterator(X_test, y_test, batchsize=batchsize)
        
        best_valid_accu = 0
        for epoch in range(max_epoch):
            print('epoch:', epoch)
            pbar = tg.ProgressBar(len(iter_train))
            ttl_train_cost = 0
            ttl_examples = 0
            print('..training')
            for X_batch, y_batch in iter_train:
                feed_dict = {X_ph:X_batch, y_ph:y_batch}
                _, train_cost = sess.run([optimizer,train_cost_sb] , feed_dict=feed_dict)              
                ttl_train_cost += len(X_batch) * train_cost
                ttl_examples += len(X_batch)
                pbar.update(ttl_examples)
            mean_train_cost = ttl_train_cost/float(ttl_examples)
            print('\ntrain cost', mean_train_cost)

            ttl_valid_cost = 0
            ttl_valid_accu = 0
            ttl_examples = 0
            pbar = tg.ProgressBar(len(iter_test))
            print('..validating')
            for X_batch, y_batch in iter_test:
                feed_dict = {X_ph:X_batch, y_ph:y_batch}
                valid_cost, valid_accu = sess.run([test_cost_sb, test_accu_sb] , feed_dict=feed_dict)
                #mask_output = sess.run(y_test_sb, feed_dict=feed_dict)
                ttl_valid_cost += len(X_batch) * valid_cost
                ttl_valid_accu += len(X_batch) * valid_accu
                ttl_examples += len(X_batch)
                pbar.update(ttl_examples)
            mean_valid_cost = ttl_valid_cost/float(ttl_examples)
            mean_valid_accu = ttl_valid_accu/float(ttl_examples)
            print('\nvalid cost', mean_valid_cost)
            print('valid accu', mean_valid_accu)
            if best_valid_accu < mean_valid_accu:
                best_valid_accu = mean_valid_accu

            if es.continue_learning(valid_error=mean_valid_cost, epoch=epoch):
                print('epoch', epoch)
                print('best epoch last update:', es.best_epoch_last_update)
                print('best valid last update:', es.best_valid_last_update)
                print('best valid accuracy:', best_valid_accu)
            else:
                print('training done!')
                break
        
        #save_path = saver.save(sess, "trained_model.ckpt")    
        #print("Model saved in file: %s" % save_path)
        
        ### 1ST PREDICTION
        predictIndex = sys.argv[1] # input from terminal
        print('Prediction 3D Scan of No #'+predictIndex)        
        intIndex = int(predictIndex)  
        
        feed_dict = {X_ph:X_test[intIndex].reshape((1,)+X_test[0].shape)}
        mask_output = sess.run(y_test_sb, feed_dict=feed_dict)

        print('mask_outpt type')        
        print(type(mask_output))
        print(mask_output.shape)        
        
        np.save('X_test_'+predictIndex+'.npy',X_test[intIndex])
        np.save('y_test_'+predictIndex+'.npy',y_test[intIndex])
        np.save('mask_output_'+predictIndex+'.npy',mask_output[0])
        
        
        ### 2ND PREDICTION
        #predictIndex = sys.argv[2] # input from terminal
        predictIndex = str(2)
        print('Prediction 3D Scan of No #'+predictIndex)        
        intIndex = int(predictIndex)  
        
        feed_dict = {X_ph:X_test[intIndex].reshape((1,)+X_test[0].shape)}
        mask_output = sess.run(y_test_sb, feed_dict=feed_dict)

        print('mask_outpt type')        
        print(type(mask_output))
        print(mask_output.shape)        
        
        np.save('X_test_'+predictIndex+'.npy',X_test[intIndex])
        np.save('y_test_'+predictIndex+'.npy',y_test[intIndex])
        np.save('mask_output_'+predictIndex+'.npy',mask_output[0])
        
        ### 3RD PREDICTION
        predictIndex = sys.argv[2] # input from terminal
        print('Prediction 3D Scan of No #'+predictIndex)        
        intIndex = int(predictIndex)  
        
        feed_dict = {X_ph:X_test[intIndex].reshape((1,)+X_test[0].shape)}
        mask_output = sess.run(y_test_sb, feed_dict=feed_dict)

        print('mask_outpt type')        
        print(type(mask_output))
        print(mask_output.shape)        
        
        np.save('X_test_'+predictIndex+'.npy',X_test[intIndex])
        np.save('y_test_'+predictIndex+'.npy',y_test[intIndex])
        np.save('mask_output_'+predictIndex+'.npy',mask_output[0])        
        