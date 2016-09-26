import tensorflow as tf


class Optimizer(object):
    def __init__(self, learning_rate=None, clipper=None):
        self.learning_rate_val = learning_rate
        self.clipper = clipper

    def build(self, learning_rate_var=None):
        if learning_rate_var is None:
            learning_rate = self.learning_rate_val
        else:
            learning_rate = learning_rate_var
        self._build(learning_rate)

    def get_train_step(self, loss_val, tvars=None):
        if tvars is None:
            tvars = tf.trainable_variables()
        grads_and_vars = self.optimizer.compute_gradients(loss_val, tvars)
        if self.clipper is not None:
            grads_and_vars = [(self.clipper.clip(g), v)
                              for (g, v) in grads_and_vars]
        return self.optimizer.apply_gradients(grads_and_vars)


class AdamOptim(Optimizer):
    def _build(self, learning_rate):
        self.optimizer = tf.train.AdamOptimizer(learning_rate)


class RMSPropOptim(Optimizer):
    def __init__(self, decay=0.95, epsilon=0.01, **kwargs):
        Optimizer.__init__(self, **kwargs)
        self.decay = decay
        self.epsilon = epsilon

    def _build(self, learning_rate):
        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=self.decay, epsilon=self.epsilon)
