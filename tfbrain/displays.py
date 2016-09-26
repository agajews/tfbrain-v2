from .helpers import avg_over_batches
import numpy as np


class Display(object):
    def __init__(self, display_freq=100, **kwargs):
        self.display_freq = display_freq

    def display_update(self, update, *args, **kwargs):
        if update % self.display_freq == 0:
            self._display_update(update, *args, **kwargs)


class LossAccDisplay(Display):
    def build(self, loss, acc):
        self.loss = loss
        self.acc = acc

    def _display_update(self, update, batch_xs, batch_y,
                        val_xs, val_y,
                        epoch, train_preprocessor, batch_size):
        display = 'Step: %d | Epoch: %d' % (update, epoch)
        train_loss = self.loss.eval(batch_xs, batch_y)
        display += ' | Train loss: %f' % train_loss
        train_acc = self.acc.eval(batch_xs, batch_y)
        display += ' | Train acc: %f' % train_acc
        if val_xs is not None and val_y is not None:
            val_acc = avg_over_batches(
                val_xs, val_y, train_preprocessor, self.acc.eval, batch_size)
            display += ' | Val acc: %f' % val_acc
        print(display)


class DQNDisplay(Display):
    def build(self, loss, net, epsilon, experience_replay):
        self.loss = loss
        self.net = net
        self.epsilon = epsilon
        self.experience_replay = experience_replay

    def _display_update(self, update):
        experience = self.experience_replay.get_most_recent()
        state, action, reward, next_state = experience
        preds = self.net.eval(state=np.expand_dims(state, 0))
        epsilon = self.epsilon.get_epsilon(update)
        print('Update: %d | Epsilon: %0.4f | Preds: %s' %
              (update, epsilon, str(preds[0])))

    def display_episode(self, reward):
        print('Episode finished with reward %0.2f' % reward)

    def display_epoch_start(self, epoch, train=False):
        epoch_text = 'Train' if train else 'Eval'
        print('-' * 40 + '%s Epoch %d' % (epoch_text, epoch) + '-' * 40)

    def display_epoch_end(self, epoch, rewards, train=False):
        epoch_text = 'Train' if train else 'Eval'
        avg_reward = sum(rewards) / len(rewards)
        print('%s epoch %d ended with avg reward %0.2f' %
              (epoch_text, epoch, avg_reward))
