from ale_python_interface.ale_python_interface import ALEInterface

# from tfbrain.helpers import bcolors

import numpy as np

import random

# import cv2

from scipy import ndimage


class AsyncAtariEnvironment(object):
    def __init__(self, hyperparams, rom_fnm):
        self.hyperparams = hyperparams
        self.show_screen = hyperparams['show_screen']
        self.state_len = hyperparams['state_len']
        # self.screen_resize = hyperparams['screen_resize']
        self.rom_fnm = rom_fnm
        self.init_ale(display=self.show_screen)
        self.actions = self.ale.getMinimalActionSet()
        print('Num possible actions: %d' % len(self.actions))
        self.state_shape = (84, 84, self.state_len)

    def init_ale(self, display=False):
        self.ale = ALEInterface()
        self.ale.setInt(b'random_seed', random.randrange(0, 999))
        # self.ale.setInt(b'delay_msec', 0)
        self.ale.setFloat(b'repeat_action_probability', 0.0)
        self.ale.setInt(b'frame_skip', self.hyperparams['frame_skip'])
        self.ale.setBool(b'color_averaging', True)
        if display:
            self.ale.setBool(b'display_screen', True)
        self.ale.loadROM(str.encode(self.rom_fnm))

    # def set_screen_shape(self):
    #     self.start_episode()
    #     # self.perform_action(0)
    #     self.state_shape = self.get_state().shape
    #     print(self.state_shape)
        # self.prev_screen_rgb = self.ale.getScreenRGB()
        # screen = self.ale.getScreenRGB()
        # screen = self.preprocess_screen(screen)
        # self.screen_shape = screen.shape
        # self.state_shape = (self.state_len,) + \
        #     self.screen_shape

    def preprocess_screen(self, screen, prev_screen):
        # screen = np.maximum(screen, prev_screen)

        # observation = cv2.cvtColor(cv2.resize(
        #     screen_rgb, (84, 110)), cv2.COLOR_BGR2GRAY)
        # observation = observation[26:110, :]
        # ret, observation = cv2.threshold(
        #     observation, 1, 255, cv2.THRESH_BINARY)
        # return np.reshape(observation, (84, 84))
        screen = np.dot(
            screen, np.array([.299, .587, .114])).astype(np.uint8)
        screen = ndimage.zoom(screen, (0.4, 0.525))
        screen.resize((84, 84))
        # screen = screen / 255.0
        return np.array(screen)

    def get_state_shape(self):
        return self.state_shape

    def start_episode(self):
        self.ale.reset_game()
        max_num = self.hyperparams['max_start_nop']
        for nop_num in range(random.randint(0, max_num)):
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
        # curr_state = np.zeros(self.state_shape)
        # for i in range(self.state_len):
        #     curr_state[i] = self.states[i]
        # return curr_state

    def perform_action(self, action_dist):
        # print(action_ind)
        action = self.actions[np.argmax(action_dist)]
        # self.curr_reward = 0
        # for frame in range(self.hyperparams['frame_skip']):
        #     self.prev_screen = self.ale.getScreenRGB()
        #     self.curr_reward += self.ale.act(action)
        self.curr_reward = self.ale.act(action)
        self.episode_reward += self.curr_reward
        # if not self.curr_reward == 0:
        #     print(bcolors.WARNING +
        #           'Got real reward!' +
        #           bcolors.ENDC)
        if self.curr_reward > 0:
            self.curr_reward = 1
            # print(bcolors.WARNING +
            #       'Got real reward!' +
            #       bcolors.ENDC)
        elif self.curr_reward < 0:
            self.curr_reward = -1
        screen = self.ale.getScreenRGB()
        screen_processed = self.preprocess_screen(screen, self.prev_screen)
        self.prev_screen = screen
        self.states = self.states[:self.state_len - 1]
        self.states.insert(0, screen_processed)

    def start_eval_mode(self):
        # self.init_ale(display=self.show_screen)
        pass

    def end_eval_mode(self):
        # self.init_ale(display=False)
        pass

    def get_reward(self):
        return self.curr_reward

    def episode_is_over(self):
        return self.ale.game_over()

    def get_actions(self):
        return list(range(len(self.actions)))


class AsyncAtariTask(object):
    def __init__(self, hyperparams, rom_fnm):
        self.hyperparams = hyperparams
        self.rom_fnm = rom_fnm
        self.prototype_environment = self.create_environment()

    def get_actions(self):
        return self.prototype_environment.get_actions()

    def get_state_shape(self):
        return self.prototype_environment.get_state_shape()

    def create_environment(self):
        return AsyncAtariEnvironment(self.hyperparams, self.rom_fnm)
