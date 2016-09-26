import blosc

from collections import deque

import random

import numpy as np


class ExperienceReplay(object):
    def __init__(self, replay_len, dtype=np.uint8, **kwargs):
        self.replay_len = replay_len
        self.state_shape = None
        self.dtype = dtype

    def __len__(self):
        return len(self.replay)

    def build(self):
        self.replay = deque(maxlen=self.replay_len)

    def clear(self):
        self.build()

    def compress(self, array):
        array = array * 255.0
        return blosc.compress(array.astype(self.dtype).flatten().tobytes(),
                              typesize=1)

    def decompress(self, compressed):
        return np.reshape(np.fromstring(
            blosc.decompress(compressed), dtype=self.dtype),
            self.state_shape) / 255.0

    def add_experience(self, experience):
        experience = self.process_experience(experience)
        self.replay.append(experience)

    def get_most_recent(self):
        return self.unprocess_experience(self.replay[-1])

    def process_experience(self, experience):
        state, action, reward, next_state = experience
        if self.state_shape is None:
            self.state_shape = state.shape
        state = self.compress(state)
        next_state = self.compress(next_state)
        return (state, action, reward, next_state)

    def unprocess_experience(self, experience):
        state, action, reward, next_state = experience
        state = self.decompress(state)
        next_state = self.decompress(next_state)
        return state, action, reward, next_state

    def sample(self, batch_size):
        indices = random.sample(range(len(self.replay)), batch_size)
        sample = []
        for index in indices:
            experience = self.unprocess_experience(self.replay[index])
            sample.append(experience)
        return sample
