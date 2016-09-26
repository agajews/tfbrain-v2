import tensorflow as tf


class Clipper(object):
    def __init__(self, **kwargs):
        pass

    def clip(self, val):
        return val


class NormClipper(object):
    def __init__(self, norm=5):
        self.norm = norm

    def clip(self, val):
        return tf.clip_by_norm(val, self.norm)
