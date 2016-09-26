from ale_python_interface.ale_python_interface import ALEInterface

import numpy as np

import random

from scipy import ndimage


class AtariTask(object):
    def __init__(self, state_len=4, frame_skip=4, max_start_nop=7,
                 rom_fnm='data/roms/breakout.bin',
                 show_screen=False):
        self.state_len = state_len
        self.frame_skip = frame_skip
        self.max_start_nop = max_start_nop
        self.rom_fnm = rom_fnm
        self.show_screen = show_screen

    def build(self):
        self.init_ale()
        self.actions = self.ale.getMinimalActionSet()
        print('Num possible actions: %d' % len(self.actions))
        self.state_shape = (84, 84, self.state_len)
        print('State shape: %s' % str(self.state_shape))

    def init_ale(self):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', random.randrange(0, 999))
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setInt(b'frame_skip', self.frame_skip)
        self.ale.setBool(b'color_averaging', True)
        self.ale.setBool(b'display_screen', self.show_screen)
        self.ale.loadROM(str.encode(self.rom_fnm))

    def preprocess_screen(self, screen, prev_screen):
        screen = np.dot(
            screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        screen.resize((84, 84))
        screen = np.array(screen)
        screen = screen / 255.0
        return screen

    def get_state_shape(self):
        return self.state_shape

    def start_episode(self):
        self.ale.reset_game()
        for nop_num in range(random.randint(0, self.max_start_nop)):
            self.ale.act(0)

        self.episode_reward = 0
        self.states = []
        screen = self.ale.getScreenRGB()
        for _ in range(self.state_len):
            prev_screen = screen
            screen = self.ale.getScreenRGB()
            self.states.append(self.preprocess_screen(screen, prev_screen))
        self.prev_screen = screen

    def get_episode_reward(self):
        return self.episode_reward

    def get_state(self):
        curr_state = np.stack(self.states, axis=2)
        return curr_state

    def perform_action(self, action):
        reward = self.ale.act(self.actions[action])
        self.episode_reward += reward
        if reward > 0:
            self.curr_reward = 1
        elif reward < 0:
            self.curr_reward = -1
        else:
            self.curr_reward = 0
        screen = self.ale.getScreenRGB()
        screen_processed = self.preprocess_screen(screen, self.prev_screen)
        self.prev_screen = screen
        self.states = self.states[:self.state_len - 1]
        self.states.insert(0, screen_processed)

    def get_reward(self):
        return self.curr_reward

    def episode_is_over(self):
        return self.ale.game_over()

    def get_actions(self):
        return self.actions
