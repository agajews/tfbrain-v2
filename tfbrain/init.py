import tensorflow as tf
import numpy as np
import math


class Init(object):
    def get_init(self, shape):
        raise NotImplementedError()


class ZeroInit(Init):
    def get_init(self, shape):
        return tf.zeros(shape)


class ConstInit(Init):
    def __init__(self, const=0.1):
        self.const = 0.1

    def get_init(self, shape):
        return tf.constant(self.const, shape=shape)


class NormalInit(Init):
    def __init__(self, mean=0.0, stddev=0.01):
        self.mean = mean
        self.stddev = stddev

    def get_init(self, shape):
        return tf.truncated_normal(shape, self.mean, self.stddev)


class PassthroughInit(Init):
    def __init__(self, var):
        self.var = var

    def get_init(self, shape):
        assert self.var.get_shape().tolist() == list(shape)
        return self.var


class PairInit(Init):
    def __init__(self, W_init=None, b_init=None):
        if W_init is None:
            W_init = NormalInit()
        if b_init is None:
            b_init = ConstInit()
        self.W_init = W_init
        self.b_init = b_init

    def get_init(self, W_shape, b_shape):
        W = self.W_init.get_init(W_shape)
        b = self.b_init.get_init(b_shape)
        return W, b


class DQNPairInit(Init):
    def get_init(self, W_shape, b_shape):
        stdv = 1.0 / math.sqrt(np.prod(W_shape[0:-1]))
        W = tf.random_uniform(W_shape, minval=-stdv, maxval=stdv)
        b = tf.random_uniform(b_shape, minval=-stdv, maxval=stdv)
        return W, b
