import gym
import numpy as np

class UpsideDownAnt(gym.Wrapper):
    """
    Modified version of Ant where the Ant starts from an upside down position
    """

    def __init__(self):
        env = gym.make('Ant-v3',
                        exclude_current_positions_from_observation=False)
        super().__init__(env)
        self.num_features = self.state_vector().shape[0]

        self.init_states = []
        while len(self.init_states) < 100:
            s = super().reset()
            step = 0
            for _ in range(5000):
                s, r, d, i = self.step(self.action_space.sample())

                s = self.state_vector()

                if step > 100 and s[2] < 0.3:
                    self.init_states.append([s[0:15], s[15:29]])
                    break

                step += 1

    def step(self, action):
        observation, reward, done, info = super().step(action)
        observation = self.state_vector()

        info['self'] = self

        return self.state_vector(), reward, done, info

    def reset(self):
        self.sim.reset()

        self.set_state(*self.init_states[np.random.randint(len(self.init_states))])

        return self.state_vector()
