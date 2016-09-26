import tensorflow as tf


class Loss(object):
    def __init__(self, **kwargs):
        pass

    def get_loss(self):
        return self.loss

    def build(self, net, y, mask=None):
        '''y_hat: Layer subclass
        y: a TF placeholder representing the expected output'''
        self.net = net
        self.y_hat = net.get_output()
        self.y = y
        self.mask = mask
        self.sess = net.get_sess()
        self._build()

    def eval(self, xs, y, mask=None):
        feed_dict = self.net.get_feed_dict(xs)
        feed_dict.update({self.y: y})
        if mask is not None:
            feed_dict.update({self.mask, mask})
        return self.sess.run(self.loss, feed_dict=feed_dict)

    def _build(self):
        '''y_hat: a TF tensor representing a model's output
        y: a TF placeholder representing the expected output'''
        raise NotImplementedError()


class MSE(Loss):
    def _build(self):
        if self.mask is None:
            errors = self.y - self.y_hat
        else:
            errors = self.y - tf.reduce_sum(self.y_hat * self.mask,
                                            reduction_indices=1)
        self.loss = tf.reduce_mean(tf.square(errors))


class MSEDQN(Loss):
    def _build(self):
        if self.mask is None:
            errors = self.y - self.y_hat
        else:
            errors = self.y - tf.reduce_sum(self.y_hat * self.mask,
                                            reduction_indices=1)
        difference = tf.abs(errors)
        quadratic_part = tf.clip_by_value(difference, 0.0, 1.0)
        linear_part = tf.sub(difference, quadratic_part)
        errors = (0.5 * tf.square(quadratic_part)) + linear_part
        self.loss = tf.reduce_sum(errors)


class Crossentropy(Loss):
    def __init__(self, log_clip=1e-10, **kwargs):
        Loss.__init__(self, **kwargs)
        self.log_clip = log_clip

    def _build(self):
        assert self.mask is None
        log_out = tf.log(tf.clip_by_value(self.y_hat, self.log_clip, 1.0))
        errors = -tf.reduce_sum(self.y * log_out, reduction_indices=1)
        self.loss = tf.reduce_mean(errors)
