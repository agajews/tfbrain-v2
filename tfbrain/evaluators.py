import tensorflow as tf


class Evaluator(object):
    def __init__(self, **kwargs):
        pass

    def build(self, net, y):
        self.net = net
        self.y = y
        self.sess = self.net.get_sess()
        self._build()

    def eval(self, xs, y):
        feed_dict = self.net.get_feed_dict(xs)
        feed_dict.update({self.y: y})
        return self.sess.run(self.accuracy, feed_dict=feed_dict)


class CategoricalEval(Evaluator):
    def _build(self):
        correct_prediction = tf.equal(tf.argmax(self.net.get_output(), 1),
                                      tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
