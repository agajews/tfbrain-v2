import tensorflow as tf

from functools import reduce

from .core import Layer


class FlattenLayer(Layer):

    def __init__(self,
                 incoming,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.output_shape = self.flatten_shape(incoming.get_output_shape())

    def get_base_name(self):
        return 'flat'

    def flatten_shape(self, shape):
        return (None,
                reduce(lambda x, y: x * y,
                       shape[1:]))

    def _get_output(self, incoming_var):
        return tf.reshape(incoming_var, (-1,) + self.output_shape[1:])


class ReshapeLayer(Layer):

    def __init__(self,
                 incoming,
                 shape,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'shape': shape})
        self.output_shape = list(map(lambda d: d if d is not None else -1,
                                     shape))

    def get_base_name(self):
        return 'reshape'

    def _get_output(self, incoming_var):
        return tf.reshape(incoming_var, self.output_shape)


class TransposeLayer(Layer):

    def __init__(self,
                 incoming,
                 perm,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'perm': perm})
        in_shape = incoming.get_output_shape()
        self.output_shape = [in_shape[i] for i in perm]
        self.perm = perm

    def get_base_name(self):
        return 'transpose'

    def _get_output(self, incoming_var):
        return tf.transpose(incoming_var, self.perm)


class SqueezeLayer(Layer):

    def __init__(self,
                 incoming,
                 dim,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'dim': dim})
        in_shape = incoming.get_output_shape()
        self.output_shape = in_shape[:dim] + in_shape[dim + 1:]
        self.dim = dim

    def get_base_name(self):
        return 'squeeze'

    def _get_output(self, incoming_var):
        return tf.squeeze(incoming_var, [self.dim])


class SeqSliceLayer(Layer):

    def __init__(self,
                 incoming,
                 col=-1,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'col': col})
        incoming_shape = incoming.get_output_shape()
        self.incoming_shape = incoming_shape
        self.output_shape = incoming_shape[:1] + \
            incoming_shape[2:]
        self.col = col

    def check_compatible(self, incoming):
        if not len(incoming.get_output_shape()) == 3:
            raise Exception(('Incoming layer\'s output shape %s '
                             'incompatible, this is only for sequences')
                            % str(incoming.get_output_shape()))

    def get_base_name(self):
        return 'slice'

    def _get_output(self, incoming_var):
        if self.col == -1:
            incoming_var = tf.reverse(
                incoming_var,
                [False, True] + [False] * (len(self.incoming_shape) - 2))
            col = 0
        else:
            col = self.col
        begin = [0] * len(self.incoming_shape)
        begin[1] = col
        size = [-1] * len(self.incoming_shape)
        size[1] = 1
        output_shape = []
        for shape in self.output_shape:
            if shape is not None:
                output_shape.append(shape)
            else:
                output_shape.append(-1)
        return tf.reshape(tf.slice(incoming_var, begin, size),
                          output_shape)


class MergeLayer(Layer):

    def __init__(self,
                 incoming_list,
                 axis=1,
                 **kwargs):
        Layer.__init__(self, incoming_list, **kwargs)
        self.config.update({'axis': axis})
        self.axis = axis
        self.output_shape = self.calc_output_shape()

    def get_base_name(self):
        return 'merge'

    def calc_output_shape(self):
        output_shape = list(self.incoming[0].get_output_shape())
        output_shape[self.axis] = -1
        axis_length = self.incoming[0].get_output_shape()[self.axis]
        for incoming in self.incoming[1:]:
            self.check_compatible(incoming, output_shape)
            axis_length += incoming.get_output_shape()[self.axis]

        output_shape[self.axis] = axis_length
        return tuple(output_shape)

    def check_compatible(self, incoming, output_shape):
        incoming_shape = list(incoming.output_shape)
        incoming_shape[self.axis] = -1
        assert incoming_shape == output_shape

    def _get_output(self, *incoming_vars):
        return tf.concat(self.axis, incoming_vars)
