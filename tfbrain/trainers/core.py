import tensorflow as tf

from tfbrain.helpers import create_minibatch_iterator


class Trainer(object):
    def __init__(self, net, loss, optim, evaluator, display, **kwargs):
        self.net = net
        self.sess = net.get_sess()
        self.optim = optim
        self.loss = loss
        self.evaluator = evaluator
        self.display = display

    def build_vars(self, target_dtype):
        self.y = tf.placeholder(target_dtype,
                                shape=self.net.get_output_shape())

    def build_loss(self):
        self.loss.build(self.net, self.y)

    def build_eval(self):
        self.evaluator.build(self.net, self.y)

    def build_display(self):
        self.display.build(self.loss, self.evaluator)

    def build_optim(self):
        self.optim.build()
        self.train_step = self.optim.get_train_step(self.loss.get_loss())

    def perform_update(self, batch_xs, batch_y):
        feed_dict = self.net.get_feed_dict(batch_xs, train=True)
        feed_dict.update({self.y: batch_y})
        self.train_step.run(feed_dict=feed_dict)

    def build(self, target_dtype=tf.float32):
        print('Building trainer ...')
        self.net.build_output()
        self.build_vars(target_dtype)
        self.build_loss()
        self.build_eval()
        self.build_optim()
        self.build_display()
        self.sess.run(tf.initialize_all_variables())

    def train(self, train_xs, train_y, val_xs=None, val_y=None,
              target_dtype=tf.float32,
              build=True, num_updates=10000, batch_size=128,
              train_preprocessor=None,
              test_preprocessor=None):
        '''train_xs: a dictionary of strings -> np arrays
        matching the model's input_vars dictionary
        train_y: a np array of expected outputs
        val_xs: same as train_xs but for validation
        val_y: same as train_y but for validation'''

        if build:
            self.build(target_dtype)

        def create_batches():
            return create_minibatch_iterator(
                train_xs, {'y': train_y},
                train_preprocessor,
                batch_size)

        minibatches = create_batches()
        epoch = 0
        for update in range(num_updates):
            try:
                batch_xs, batch_y = next(minibatches)
            except StopIteration:
                minibatches = create_batches()
                batch_xs, batch_y = next(minibatches)
                epoch += 1
            self.perform_update(batch_xs, batch_y['y'])
            self.display.display_update(
                update, batch_xs, batch_y['y'],
                val_xs, val_y,
                epoch, test_preprocessor, batch_size)
