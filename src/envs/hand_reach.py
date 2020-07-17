import gym
import numpy as np


class HandReach(gym.Wrapper):

    def __init__(self):
        env = gym.make('HandReach-v0')
        super().__init__(env)
        self.num_features = self.observation_space.sample()['observation'].shape[0]

    def seed(self, seed=None):
        return super().seed(seed)

    def step(self, action):
        obs_dict, reward, done, info = super().step(action)
        return obs_dict['observation'], reward, done, info

    def reset(self):
        obs_dict = super().reset()
        return obs_dict['observation']

    def render(self, mode='human'):
        return super().render(mode)
