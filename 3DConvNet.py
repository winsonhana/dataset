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
from tensorgraph.cost import entropy, accuracy
from math import ceil
import WMH_loadData  # 3D MRI Scanned Dataset


class Conv3D_Tranpose1():
    def __init__(self, input_channels, num_filters, output_shape, kernel_size=(3,3,3), stride=(1,1,1),
                 filter=None, b=None, padding='VALID'):

        self.input_channels = input_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_shape = output_shape
        
        self.filter_shape = self.kernel_size + (self.num_filters, self.input_channels)
        self.filter = filter
        if self.filter is None:
            self.filter = tf.Variable(tf.random_normal(self.filter_shape, stddev=0.1),
                                      name=self.__class__.__name__ + '_filter')

        self.b = b
        if self.b is None:
            self.b = tf.Variable(tf.zeros([self.num_filters]), name=self.__class__.__name__ + '_b')


    def _train_fprop(self, state_below):
        '''
        state_below: (b, d, h, w, c)
        '''
        batch_size = tf.shape(state_below)[0]
        depth, height, width = self.output_shape
        deconv_shape = tf.stack((batch_size, int(depth), int(height), int(width), self.num_filters))
        conv_out = tf.nn.conv3d_transpose(value=state_below, filter=self.filter, output_shape=deconv_shape,
                                          strides=(1,)+self.stride+(1,), padding=self.padding)
        return tf.nn.bias_add(conv_out, self.b)
        
    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)

    @property
    def _variables(self):
        return [self.filter, self.b]


class MaxPool3D():
    def __init__(self, poolsize=(2,2,2), stride=(1,1,1), padding='VALID'):
        self.poolsize = (1,) + poolsize + (1,)
        self.stride = (1,) + stride + (1,)
        self.padding = padding

    def _train_fprop(self, state_below):
        return tf.nn.max_pool3d(state_below, ksize=self.poolsize,
                              strides=self.stride, padding=self.padding, name=None)

    def _test_fprop(self, state_below):
        return self._train_fprop(state_below)


def updateConvLayerSize(dataDimension,stride):
    assert len(dataDimension) == len(stride), "TensorRank of dataset is not the same as stride's rank."
    output_ = tuple()
    for i in range(len(stride)):
        output_ += (int(ceil(dataDimension[i]/float(stride[i]))),)
    return output_
    

def model3D(img=(48,240,240)):
    with tf.name_scope('WMH'):
        seq = tg.Sequential()
        convStride = (1,1,1)
        poolStride = (2,2,2)
        #poolStride = (1,1,1)
        seq.add(Conv3D(input_channels=1, num_filters=8, kernel_size=(3,3,3), stride=convStride, padding='SAME'))        
        #seq.add(TFBatchNormalization(name='b1'))
        seq.add(MaxPool3D(poolsize=(2,2,2), stride=poolStride, padding='SAME'))
        layerSize1 = updateConvLayerSize(img,poolStride)
        print("layer1: "+str(layerSize1))
        seq.add(RELU())
        seq.add(Conv3D(input_channels=8, num_filters=16, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        #seq.add(TFBatchNormalization(name='b2'))
        layerSize2 = updateConvLayerSize(layerSize1,convStride)
        print("layer1: "+str(layerSize2))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=16, num_filters=8, output_shape=layerSize1, kernel_size=(3,3,3), stride=convStride, padding='SAME'))
        seq.add(RELU())
        seq.add(Conv3D_Tranpose1(input_channels=8, num_filters=1, output_shape=img, kernel_size=(3,3,3), stride=(2,2,2), padding='SAME'))
        seq.add(Sigmoid())
    return seq
        

if __name__ == '__main__':

    learning_rate = 0.001
    batchsize = 1

    max_epoch = 10
    es = tg.EarlyStopper(max_epoch=max_epoch,
                         epoch_look_back=3,
                         percent_decrease=0)


    seq = model3D()
    print("MODEL INIT")
    dataset = WMH_loadData.WMHdataset('./dataset')
    assert dataset.AbleToRetrieveData(), 'not able to locate the directory of dataset'
    dataset.InitDataset(split=1.0)         # Take everything
#    dataX, dataY = dataset.NextBatch3D(20) # Take everything
#    X_train = dataX[:15]
#    X_test = dataX[15:]
#    y_train = dataY[:15]
#    y_test = dataY[15:]
#    #X_train, y_train, X_test, y_test = Mnist(flatten=False, onehot=True, binary=True, datadir='.')
#    iter_train = tg.SequentialIterator(X_train, y_train, batchsize=batchsize)
#    iter_test = tg.SequentialIterator(X_test, y_test, batchsize=batchsize)


    X_ph = tf.placeholder('float32', [None, 48,240,240,1])
    y_ph = tf.placeholder('float32', [None, 48,240,240,1])
    print("PLACEHOLDER")

    y_train_sb = seq.train_fprop(X_ph)
    print("train_fprop")
    y_test_sb = seq.test_fprop(X_ph)
    print("test_fprop")
    #train_cost_sb = tf.reduce_mean((y_ph - y_train_sb)**2)
    train_cost_sb = entropy(y_ph, y_train_sb)
    print("entropy1")
    #test_cost_sb = tf.reduce_mean((y_ph - y_test_sb)**2)
    test_cost_sb = entropy(y_ph, y_test_sb)
    print("entropy2")
    test_accu_sb = accuracy(y_ph, y_test_sb)
    print("entropy3")
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(train_cost_sb)
    print("OPTIMIZER")
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    
    with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
        init = tf.global_variables_initializer()
        print("VAR INIT")
        sess.run(init)
        print("SESSION INIT")
        
        batchsize = 1
        dataX, dataY = dataset.NextBatch3D(20) # Take everything
        print("retreive data from HDD")
        X_train = dataX[:15]
        X_test = dataX[15:]
        y_train = dataY[:15]
        y_test = dataY[15:]
        #X_train, y_train, X_test, y_test = Mnist(flatten=False, onehot=True, binary=True, datadir='.')
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
                print("feed")
                _, train_cost = sess.run([optimizer,train_cost_sb] , feed_dict=feed_dict)
                print("cost")                
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




