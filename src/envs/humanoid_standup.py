import gym
import numpy as np


class HumanoidStandup(gym.Wrapper):
    """
    Humanoid standup with standard humanoid mechanics
    """

    def __init__(self):
        env = gym.make('Humanoid-v3')
        super().__init__(env)
        self.num_features = self.state_vector().shape[0]

        self.init_states = []
        while len(self.init_states) < 100:
            s = super().reset()
            step = 0
            for _ in range(5000):
                s, r, d, i = self.step(self.action_space.sample())

                s = self.state_vector()

                if step > 100 and s[2] < 1.0:
                    self.init_states.append([s[0:24], s[24:47]])
                    break

                step += 1

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.state_vector()
        return self.state_vector(), reward, done, info

    def reset(self):
        self.sim.reset()

        self.set_state(*self.init_states[np.random.randint(len(self.init_states))])

        return self.state_vector()


