import tensorflow as tf

from .core import Layer


class DropoutLayer(Layer):

    def __init__(self,
                 incoming,
                 keep_prob,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'keep_prob': keep_prob})
        self.output_shape = incoming.get_output_shape()
        self.keep_prob = keep_prob
        self.prob_var = tf.placeholder(tf.float32)

    def get_base_name(self):
        return 'drop'

    def _get_supp_feed_dict(self, train=False):
        if train:
            return {self.prob_var: self.keep_prob}
        else:
            return {self.prob_var: 1.0}

    def _get_output(self, incoming_var, **kwargs):
        return tf.nn.dropout(incoming_var, self.prob_var)


class ScaleLayer(Layer):

    def __init__(self, incoming, scale, **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'scale': scale})
        self.scale = scale
        self.output_shape = incoming.get_output_shape()

    def get_base_name(self):
        return 'scale'

    def _get_output(self, incoming_var, **kwargs):
        return tf.to_float(incoming_var) * self.scale
