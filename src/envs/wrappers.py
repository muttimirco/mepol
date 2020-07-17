import gym


class ErgodicEnv(gym.Wrapper):
    """
    Environment wrapper to ignore the done signal
    assuming the MDP is ergodic.
    """
    def __init__(self, env):
        super().__init__(env)

    def step(self, a):
        s, r, d, i = super().step(a)
        d = False
        return s, r, d, i

class CoreStateEnv(gym.Wrapper):
    """
    Environment wrapper to filter a subset of the full state space.
    This can be useful in Mujoco to recover the "core" state
    which is normally obtained using the env.env.state_vector() call.
    """
    def __init__(self, env, features_range):
        super().__init__(env)
        self.features_range = features_range
        self.num_features = len(features_range)

    def core_state(self, s):
        return s[self.features_range]

    def reset(self):
        s = super().reset()
        return self.core_state(s)

    def step(self, a):
        s, r, d, i = super().step(a)
        s = self.core_state(s)
        return s, r, d, i

class CustomRewardEnv(gym.Wrapper):
    """
    Environment wrapper to provide a custom reward function
    and episode termination.
    """
    def __init__(self, env, get_reward):
        super().__init__(env)
        self.get_reward = get_reward

    def step(self, a):
        s, r, d, i = super().step(a)
        r, d = self.get_reward(s, r, d, i)
        return s, r, d, i
