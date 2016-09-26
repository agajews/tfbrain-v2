class EpsilonAnnealer(object):
    def __init__(self, start=1.0, end=0.1, sep=1000000, **kwargs):
        self.start = start
        self.end = end
        self.sep = sep

    def build(self):
        pass

    def get_epsilon(self, update):
        if update <= self.sep:
            return self.start + update * (self.end - self.start) / self.sep
        else:
            return self.end
