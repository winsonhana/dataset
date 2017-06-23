import tensorflow as tf
from tensorgraph.layers import Conv3D, RELU

class SoftMaxMultiDim():
    """Computes softmax activations for multidimensional Tensor
    
      softmax = exp(Tensor) / reduce_sum(exp(Tensor), dim)
      where reduce_sum can be done across multi dimension as well.
      E.g. a Tensor of shape (a,b,c,d,e,f), softmax can compute across an array
      of dimension = [b,c,f]

    """
    def __init__(self, axis=-1):
        self.axis = axis
    def softmaxDim(target, axis):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target-max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = target_exp / normalize
        return softmax
    def _train_fprop(self, state_below):
        return self.softmaxDim(state_below, self.axis)    
    def _test_fprop(self, state_below):
        return self._train_fprop(self, state_below)



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


##############################
class Residual3D():

    def __init__(self, input, num_blocks, kernel=(5,5,5)):
        self.num_blocks = num_blocks
        self.input = input
        self.kernel = kernel
        self.blocks = []
        for _ in range(self.num_blocks):
            layers = []
            layers.append(Conv3D(input_channels=self.input, num_filters=self.input, kernel_size=self.kernel, stride=(1,1,1), padding='SAME'))
            layers.append(RELU())
            layers.append(Conv3D(input_channels=self.input, num_filters=self.input, kernel_size=self.kernel, stride=(1,1,1), padding='SAME'))
            self.blocks.append(layers)
        #self.blocks.append([Conv3D(input_channels=self.input, num_filters=self.input, kernel_size=self.kernel, stride=(1,1,1), padding='SAME')])

    def _train_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._train_fprop(out)
            state_below = out + state_below
            tf.nn.relu(state_below)     # RELU after adding residual block
        return state_below


    def _test_fprop(self, state_below):
        for block in self.blocks:
            out = state_below
            for layer in block:
                out = layer._test_fprop(out)
            state_below = out + state_below
            tf.nn.relu(state_below)     # RELU after adding residual block
        return state_below

###############################
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
        
    def _variables(self):
        return self.poolsize
        


#####################
def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual: 
        with tf.variable_scope('shared_activation'):
            x = self._batch_norm('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)
            orig_x = x
    else:
        with tf.variable_scope('residual_only_activation'):
            orig_x = x
            x = self._batch_norm('init_bn', x)
            x = self._relu(x, self.hps.relu_leakiness)

    with tf.variable_scope('sub1'):
        x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
        x = self._batch_norm('bn2', x)
        x = self._relu(x, self.hps.relu_leakiness)
        x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
        if in_filter != out_filter:
            orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
            orig_x = tf.pad(
                orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
        x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x