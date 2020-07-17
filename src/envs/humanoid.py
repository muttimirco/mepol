import numpy as np
import gym


class Humanoid(gym.Wrapper):

    def __init__(self):
        hum_env = gym.make('Humanoid-v3',
                           exclude_current_positions_from_observation=False)
        super().__init__(hum_env)
        self.num_features = self.state_vector().shape[0]

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return self.state_vector(), reward, done, info

    def reset(self):
        super().reset()
        return self.state_vector()

