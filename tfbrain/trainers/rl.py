import tensorflow as tf

import random

import numpy as np


class DQNTrainer(object):
    def __init__(self, net, optim, loss, task, epsilon,
                 experience_replay, display, saver,
                 train_freq=4, eval_epsilon=0.05, target_update_freq=40000,
                 batch_size=32, reward_discount=0.99,
                 load_params=False, **kwargs):
        self.net = net
        self.sess = self.net.get_sess()
        self.optim = optim
        self.loss = loss
        self.task = task
        self.epsilon = epsilon
        self.experience_replay = experience_replay
        self.display = display
        self.saver = saver
        self.train_freq = train_freq
        self.eval_epsilon = eval_epsilon
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.reward_discount = reward_discount
        self.load_params = load_params

    def build_target(self):
        self.target_net = self.net.clone()
        self.target_net.build_output()
        self.update_target_ops = self.target_net.set_all_params_from(self.net)

    def update_target(self):
        self.sess.run(self.update_target_ops)

    def build_task(self):
        self.task.build()
        self.actions = self.task.get_actions()

    def build_vars(self):
        self.y = tf.placeholder(dtype=tf.float32, shape=(None,))
        self.mask = tf.placeholder(shape=(None, len(self.actions)),
                                   dtype=tf.float32)

    def build_loss(self):
        self.loss.build(self.net, self.y, self.mask)

    def build_optim(self):
        self.optim.build()
        tvars = self.net.get_all_params(flatten=True)
        self.train_step = self.optim.get_train_step(
            self.loss.get_loss(), tvars)

    def build_display(self):
        self.display.build(self.loss, self.net,
                           self.epsilon, self.experience_replay)

    def build_epsilon(self):
        self.epsilon.build()

    def build_replay(self):
        self.experience_replay.build()

    def build(self):
        self.net.build_output()
        self.build_target()
        self.build_task()
        self.build_vars()
        self.build_loss()
        self.build_optim()
        self.build_display()
        self.build_epsilon()
        self.build_replay()
        self.sess.run(tf.initialize_all_variables())

    def choose_action(self, state, train=False, update=None, exploring=False):
        if exploring:
            epsilon = 1.0
        elif train:
            epsilon = self.epsilon.get_epsilon(update)
        else:
            epsilon = self.eval_epsilon
        if random.random() < epsilon:
            return random.randrange(len(self.actions))
        else:
            return np.argmax(self.net.eval(state=np.expand_dims(state, 0)))

    def do_experience(self, train=False, update=None, exploring=False):
        state = self.task.get_state()
        action = self.choose_action(state, train, update, exploring)
        self.task.perform_action(action)
        reward = self.task.get_reward()
        next_state = self.task.get_state()
        return state, action, reward, next_state

    def can_perform_update(self):
        return len(self.experience_replay) >= self.batch_size

    def perform_update(self):
        experiences = self.experience_replay.sample(self.batch_size)
        x = np.array([s for (s, a, r, n) in experiences])
        y = np.zeros((self.batch_size,))
        mask = np.zeros((self.batch_size, len(self.actions)))
        preds = self.target_net.eval(
            state=np.array([n for (s, a, r, n) in experiences]))
        for experience_num, experience in enumerate(experiences):
            state, action, reward, next_state = experience
            futures = np.max(preds[experience_num])
            y[experience_num] = reward + self.reward_discount * futures
            mask[experience_num, action] = 1
        feed_dict = self.net.get_feed_dict({'state': x}, train=True)
        feed_dict.update({self.mask: mask,
                          self.y: y})
        self.sess.run(self.train_step, feed_dict=feed_dict)

    def run_train_epoch(self, start_update, num_updates,
                        exploring=False, perform_updates=True):
        rewards = []
        self.task.start_episode()
        for update in range(start_update, start_update + num_updates):
            if self.task.episode_is_over():
                reward = self.task.get_episode_reward()
                rewards.append(reward)
                self.display.display_episode(reward)
                self.task.start_episode()
            experience = self.do_experience(
                train=True, update=update, exploring=exploring)
            self.experience_replay.add_experience(experience)
            if update % self.target_update_freq == 0:
                self.update_target()
            if perform_updates and \
               update % self.train_freq == 0 and \
               self.can_perform_update():
                self.perform_update()
            self.display.display_update(update)
        return rewards

    def run_eval_epoch(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            self.task.start_episode()
            while not self.task.episode_is_over():
                self.do_experience()
            reward = self.task.get_episode_reward()
            self.display.display_episode(reward)
            rewards.append(reward)
        return rewards

    def train(self, num_updates=10000000, updates_per_epoch=50000,
              episodes_per_eval=32, init_explore_len=50000, build=True):
        if build:
            self.build()
        if self.load_params:
            self.saver.load()
        num_epochs = round(num_updates / updates_per_epoch)
        update = 0
        self.run_train_epoch(update, init_explore_len, perform_updates=False)
        for epoch in range(num_epochs):
            self.saver.save()
            self.display.display_epoch_start(epoch, train=True)
            train_rewards = self.run_train_epoch(update, updates_per_epoch)
            update += updates_per_epoch
            self.display.display_epoch_end(epoch, train_rewards, train=True)
            self.display.display_epoch_start(epoch, train=False)
            eval_rewards = self.run_eval_epoch(episodes_per_eval)
            self.display.display_epoch_end(epoch, eval_rewards, train=False)
