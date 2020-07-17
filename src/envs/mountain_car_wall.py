import numpy as np
import math

from gym.envs.classic_control.continuous_mountain_car import Continuous_MountainCarEnv


class MountainCarContinuous(Continuous_MountainCarEnv):

    def __init__(self):
        super().__init__()
        self.num_features = 2

    def step(self, action):
        position = self.state[0]
        velocity = self.state[1]
        force = min(max(action[0], -1.0), 1.0)

        velocity += force*self.power -0.0025 * math.cos(3*position)
        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed
        position += velocity
        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        if position > self.goal_position:
            # Clip position to goal position
            position = self.goal_position

            # Either set velocity to 0 or bounce back with a negative velocity
            velocity = 0.0
            # velocity = -0.01

        done = bool(position >= self.goal_position and velocity >= self.goal_velocity)

        # Make the environment non episodic
        done = False

        reward = 0
        if done:
            reward = 100.0
        reward-= math.pow(action[0],2)*0.1

        self.state = np.array([position, velocity])
        return self.state, reward, done, {}