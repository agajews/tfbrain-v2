import tensorflow as tf

from tfbrain import nonlin, init
from .core import Layer


def conv_output_length(input_length,
                       filter_length,
                       stride,
                       pad):
    if pad == 'VALID':
        output_length = input_length - filter_length + 1
    elif pad == 'SAME':
        output_length = input_length
    else:
        raise Exception('Invalid padding algorithm %s' % pad)

    return (output_length + stride - 1) // stride


class Conv2DLayer(Layer):

    def __init__(self,
                 incoming,
                 filter_size,
                 num_filters,
                 inner_strides=(1, 1),
                 pad='SAME',
                 initializer=None,
                 nonlinearity=nonlin.relu,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'filter_size': filter_size,
                            'num_filters': num_filters,
                            'inner_strides': inner_strides,
                            'pad': pad,
                            'nonlinearity': nonlinearity,
                            'initializer': initializer})

        self.nonlin = nonlinearity
        num_channels = incoming.get_output_shape()[3]
        self.output_shape = self.calc_output_shape(incoming,
                                                   filter_size,
                                                   num_filters,
                                                   num_channels,
                                                   inner_strides,
                                                   pad)
        self.strides = (1,) + inner_strides + (1,)
        self.pad = pad

        if initializer is None:
            initializer = init.PairInit()
        W_shape = filter_size + (num_channels, num_filters)
        b_shape = (num_filters,)

        W_init, b_init = initializer.get_init(W_shape, b_shape)
        self.W = tf.Variable(W_init, trainable=self.trainable)
        self.b = tf.Variable(b_init, trainable=self.trainable)

        self.params = {'W': self.W,
                       'b': self.b}

    def get_base_name(self):
        return 'conv2d'

    def calc_output_shape(self,
                          incoming,
                          filter_size,
                          num_filters,
                          num_channels,
                          inner_strides,
                          pad):

        input_shape = incoming.get_output_shape()
        output_height = conv_output_length(input_shape[1],
                                           filter_size[0],
                                           inner_strides[0],
                                           pad)
        output_width = conv_output_length(input_shape[2],
                                          filter_size[1],
                                          inner_strides[1],
                                          pad)
        return (None,
                output_height, output_width,
                num_filters)

    def _get_output(self, incoming_var):
        conv = tf.nn.conv2d(incoming_var,
                            self.W,
                            strides=self.strides,
                            padding=self.pad)
        return self.nonlin(conv + self.b)
