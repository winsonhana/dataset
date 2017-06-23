# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 12:24:48 2017

@author: winsoncws
"""
import sys
import tensorgraph as tg
import tensorflow as tf
from tensorgraph.cost import entropy, accuracy, iou, smooth_iou, image_f1
from WMH_loadT1Flair import WMHdataset # 3D MRI Scanned Dataset
#from conv3D import Conv3D_Tranpose1, MaxPool3D
#import matplotlib.pyplot as plt
import WMH_model3D # all model
#from scipy.misc import imsave
import numpy as np

if __name__ == '__main__':


    learning_rate = 0.001
    #batchsize = 6
    split = 48 # Train Valid Split
    

    max_epoch = 50
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)


    seq = WMH_model3D.model3D_Residual()
    dataset = WMHdataset('./WMH')
    assert dataset.AbleToRetrieveData(), 'not able to locate the directory of dataset'
    
    X_ph = tf.placeholder('float32', [None, 83, 256, 256, 2]) 
    y_ph = tf.placeholder('uint8', [None, 83, 256, 256, 1])
    y_ph_cat = tf.one_hot(y_ph,3) # --> unstack into 3 categorical Tensor [?, 83, 256, 256, 1, 3]
    y_ph_cat = tf.reduce_max(y_ph_cat, 4)   # --> collapse the extra 4th redundant dimension

    #### COST FUNCTION
    y_train_sb = seq.train_fprop(X_ph)
    y_test_sb = seq.test_fprop(X_ph)

    train_cost_background = (1 - smooth_iou(y_ph_cat[:,:,:,:,0] , y_train_sb[:,:,:,:,0]) )*0.1
    train_cost_label = (1 - smooth_iou(y_ph_cat[:,:,:,:,1] , y_train_sb[:,:,:,:,1]) )*0.6
    train_cost_others = (1 - smooth_iou(y_ph_cat[:,:,:,:,2] , y_train_sb[:,:,:,:,2]) )*0.4
    train_cost_average = tf.reduce_mean([train_cost_background,train_cost_label,train_cost_others])

    valid_cost_background = (1 - smooth_iou(y_ph_cat[:,:,:,:,0] , y_test_sb[:,:,:,:,0]) )*0.1
    valid_cost_label = (1 - smooth_iou(y_ph_cat[:,:,:,:,1] , y_test_sb[:,:,:,:,1]) )*0.6
    valid_cost_others = (1 - smooth_iou(y_ph_cat[:,:,:,:,2] , y_test_sb[:,:,:,:,2]) )*0.4
    valid_cost_average = tf.reduce_mean([valid_cost_background,valid_cost_label,valid_cost_others])  

    # Accuracy
    test_accu_sb = iou(y_ph_cat, y_test_sb, threshold=0.5)

    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_average)

    # model Saver
    saver = tf.train.Saver()
    
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        print("INITIALIZE SESSION")

        dataset.InitDataset(splitRatio=0.80, shuffle=False)  # Take everything 100%
        
        batchsize = 3
        #######
        # Just to train 0 & 1, ignore 2=Other Pathology. Assign 2-->0
        # dataY[dataY ==2] = 0
        #######
        X_train, y_train = dataset.NextBatch3D(len(dataset.listTrain),dataset='train')
        X_test, y_test = dataset.NextBatch3D(len(dataset.listValid),dataset='validation')

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
                _, train_cost = sess.run([optimizer,train_cost_average] , feed_dict=feed_dict)              
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
                valid_cost, valid_accu = sess.run([valid_cost_average, test_accu_sb] , feed_dict=feed_dict)
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
        
        save_path = saver.save(sess, "trained_model_2.ckpt")    
        print("Model saved in file: %s" % save_path)
        
        # PREDICTION
        predictIndex = sys.argv[1] # input from terminal
        #predictIndex = 6
        intIndex = int(predictIndex)
        print('Predicting Scanner No#'+predictIndex)        
        
        feed_dict = {X_ph:X_test[intIndex].reshape((1,)+X_test[0].shape)}
#       valid_cost, valid_accu = sess.run([test_cost_sb, test_accu_sb] , feed_dict=feed_dict)
        mask_output = sess.run(y_test_sb, feed_dict=feed_dict)

        print('mask_outpt type')        
        print(type(mask_output))
        #mask_output = (mask_output > 0.5).astype(int)
        #mask_output = mask_output * 255.0
        print(mask_output.shape)        
        

        np.save('X_test_'+predictIndex+'.npy',X_test[intIndex])
        np.save('y_test_'+predictIndex+'.npy',y_test[intIndex])
        np.save('mask_output_'+predictIndex+'.npy',mask_output[0])

        
        ####### Plotting
#        slice = 47    
#        imageTOP = np.concatenate((X_test[predictIndex,slice,:,:,0],y_test[predictIndex,slice,:,:,0]),axis=1)
#        imageBOT = np.concatenate((mask_output[0,slice,:,:,0],mask_output[0,slice,:,:,0]),axis=1)
#        #imageBOT = np.concatenate((X_test[predictIndex,slice,:,:,0],y_test[predictIndex,slice,:,:,0]),axis=1)  
#        images = np.concatenate((imageTOP,imageBOT), axis=0)
#        imsave('predictMask'+str(slice)+'.png', images)
#        
#        imsave('training_pic.png',y_test[6,48,:,:,0])        
#        imsave('training_pic2.png',imageTOP)        
        

#        print('predict Object %d of cross-section :' % predictIndex, (slice))
#        cmap_ = 'CMRmap'
#        plt.figure(figsize=(7,7))
#        plt.subplot(2,2,1)
#        plt.imshow(X_test[predictIndex,slice,:,:,0], cmap_)
#        plt.title('Flair Image')
#        plt.subplot(2,2,2)
#        plt.imshow(mask_output[predictIndex,slice,:,:,0], cmap_)
#        plt.title('Predicted Mask, accuracy: %d' % valid_accu)
#        plt.subplot(2,2,3)
#        plt.imshow(y_test[predictIndex,slice,:,:,0], cmap_)
#        plt.title('Actual Mask')
#        plt.tight_layout()
#        fig = plt.gcf() # setup png saving file
#        fig.set_size_inches(5, 5)
#        fig.savefig('predictMask'+str(slice)+'.png', dpi=200)
        


