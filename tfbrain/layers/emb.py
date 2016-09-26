import tensorflow as tf

from tfbrain import init
from .core import Layer


class EmbeddingLayer(Layer):
    def __init__(self,
                 incoming,
                 num_nodes,
                 num_cats,
                 initializer=None,
                 **kwargs):
        Layer.__init__(self, [incoming], **kwargs)
        self.config.update({'num_nodes': num_nodes,
                            'num_cats': num_cats,
                            'initializer': initializer})
        self.num_nodes = num_nodes
        self.output_shape = incoming.get_output_shape() + (num_nodes,)

        if initializer is None:
            initializer = init.PairInit()
        E_shape = (num_cats, num_nodes)
        E_init = initializer.get_init(E_shape)
        self.E = tf.Variable(E_init, trainable=self.trainable)
        self.params = {'E': self.E}

    def get_base_name(self):
        return 'emb'

    def _get_output(self, incoming_var, **kwargs):
        return tf.nn.embedding_lookup(self.E, incoming_var)
