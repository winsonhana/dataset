import tensorflow as tf

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