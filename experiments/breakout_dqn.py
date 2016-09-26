import tfbrain as tb

from tasks.rl import AtariTask

import tensorflow as tf


def train():
    task = AtariTask()
    task.build()
    sess = tf.InteractiveSession()
    net = tb.ly.InputLayer((None,) + task.get_state_shape(), sess,
                           name='state')
    net = tb.ly.Conv2DLayer(net, (8, 8), 32, inner_strides=(4, 4),
                            initializer=tb.init.DQNPairInit(),
                            pad='VALID')
    net = tb.ly.Conv2DLayer(net, (4, 4), 64, inner_strides=(2, 2),
                            initializer=tb.init.DQNPairInit(),
                            pad='VALID')
    net = tb.ly.Conv2DLayer(net, (3, 3), 64, inner_strides=(1, 1),
                            initializer=tb.init.DQNPairInit(),
                            pad='VALID')
    net = tb.ly.FlattenLayer(net)
    net = tb.ly.FullyConnectedLayer(net, 512,
                                    initializer=tb.init.DQNPairInit())
    net = tb.ly.FullyConnectedLayer(net, len(task.get_actions()),
                                    initializer=tb.init.DQNPairInit(),
                                    nonlinearity=tb.nonlin.identity)
    loss = tb.MSEDQN()
    optim = tb.RMSPropOptim(learning_rate=0.00025)
    epsilon = tb.EpsilonAnnealer(1.0, 0.1, 1000000)
    experience_replay = tb.ExperienceReplay(1000000)
    saver = tb.Saver(net, 'params/breakout_dqn.json')
    display = tb.DQNDisplay()
    trainer = tb.DQNTrainer(net, optim, loss, task, epsilon,
                            experience_replay, display, saver,
                            target_update_freq=40000)
    trainer.train(updates_per_epoch=50000, episodes_per_eval=32,
                  init_explore_len=50000)

if __name__ == '__main__':
    train()
