import tfbrain as tb
import tensorflow as tf

from tasks.mnist import load_data


def train():
    sess = tf.InteractiveSession()
    net = tb.ly.InputLayer((None, 784), sess, name='image')
    net = tb.ly.ReshapeLayer(net, shape=(None, 28, 28, 1))
    net = tb.ly.Conv2DLayer(net, (5, 5), 32)
    net = tb.ly.MaxPool2DLayer(net, (2, 2), inner_strides=(2, 2))
    net = tb.ly.Conv2DLayer(net, (5, 5), 64)
    net = tb.ly.MaxPool2DLayer(net, (2, 2), inner_strides=(2, 2))
    net = tb.ly.FlattenLayer(net)
    net = tb.ly.FullyConnectedLayer(net, 1024)
    net = tb.ly.DropoutLayer(net, 0.5)
    net = tb.ly.FullyConnectedLayer(net, 10, nonlinearity=tb.nonlin.softmax)

    loss = tb.Crossentropy(grad_norm_clip=5)
    evaluator = tb.CategoricalEval()
    display = tb.LossAccDisplay(display_freq=100)
    optim = tb.AdamOptim(learning_rate=1e-4)
    trainer = tb.Trainer(net, loss, optim, evaluator, display)

    mnist = load_data()

    train_xs = {'image': mnist['train']['images']}
    train_y = mnist['train']['labels']
    val_xs = {'image': mnist['test']['images']}
    val_y = mnist['test']['labels']

    trainer.train(train_xs, train_y,
                  val_xs, val_y,
                  batch_size=50,
                  num_updates=20000)


if __name__ == '__main__':
    train()
