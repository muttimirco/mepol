import gym
import numpy as np

# Suppress pygame welcome print
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame


class BoundingBox:
    """
    2d bounding box.
    """
    def __init__(self, xmin, xmax, ymin, ymax):
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    def check_if_inside(self, x, y):
        """
        Checks whether point (x,y) is inside the bounding box.
        """
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def rect(self, tx, ty, scale):
        """
        Returns a ready to render rect given translation and scale factors: (top left coordinates, width, height).
        """
        return (self.xmin*scale + tx, self.ymin*scale + ty, (self.xmax - self.xmin)*scale, (self.ymax - self.ymin)*scale)


class GridWorldContinuous(gym.Env):

    def __init__(self, dim=6, max_delta=0.2, wall_width=2.5):

        self.num_features = 2

        # The gridworld bottom left corner is at (-self.dim, -self.dim)
        # and the top right corner is at (self.dim, self.dim)
        self.dim = dim

        # The maximum change in position obtained through an action
        self.max_delta = max_delta

        # Maximum (dx,dy) action
        self.max_action = np.array([self.max_delta, self.max_delta], dtype=np.float32)
        self.action_space = gym.spaces.Box(-self.max_action, self.max_action, dtype=np.float32)

        # Maximum (x,y) position
        self.max_position = np.array([self.dim, self.dim], dtype=np.float32)
        self.observation_space = gym.spaces.Box(-self.max_position, self.max_position, dtype=np.float32)

        # Current state
        self.state = None

        # Initial state
        init_top_left = np.array([-self.dim, -self.dim], dtype=np.float32)
        init_bottom_right = np.array([-self.dim+2, -self.dim+2], dtype=np.float32)
        self.init_states = gym.spaces.Box(init_top_left, init_bottom_right, dtype=np.float32)

        self.wall_width = wall_width

        # The area in which the agent can move is defined by the gridworld box (defined by self.dim) and by the following walls
        self.walls = [
            # Central walls
            BoundingBox(xmin=-self.wall_width/2, xmax=self.wall_width/2, ymin=-self.wall_width, ymax=self.wall_width),
            BoundingBox(xmin=-self.wall_width, xmax=-self.wall_width/2, ymin=-self.wall_width/2, ymax=self.wall_width/2),
            BoundingBox(xmin=self.wall_width/2, xmax=self.wall_width, ymin=-self.wall_width/2, ymax=self.wall_width/2),
            # External walls
            BoundingBox(xmin=-self.dim, xmax=-(self.dim - self.wall_width), ymin=-self.wall_width/2, ymax=self.wall_width/2),
            BoundingBox(xmin=-self.wall_width/2, xmax=self.wall_width/2, ymin=-self.dim, ymax=-(self.dim - self.wall_width)),
            BoundingBox(xmin=self.dim - self.wall_width, xmax=self.dim, ymin=-self.wall_width/2, ymax=self.wall_width/2),
            BoundingBox(xmin=-self.wall_width/2, xmax=self.wall_width/2, ymin=self.dim-self.wall_width, ymax=self.dim)
        ]

        # Render stuff
        self.game_display = None
        self.DISPLAY_WIDTH = 800
        self.DISPLAY_HEIGHT = 600
        self.SCALE = 30
        self.BLUE = (0, 0, 255)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.AGENT_RADIUS = 5

    def reset(self):
        # Start near the bottom left corner of the bottom left room
        self.state = self.init_states.sample()

        # Reset pygame
        self.game_display = None

        return self.state

    def render(self):
        if self.game_display is None:
            pygame.init()
            self.game_display = pygame.display.set_mode((self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT))
            pygame.display.set_caption('Continuous Gridworld')

        # Draw background
        self.game_display.fill(self.WHITE)

        # Draw walls
        for bbox in self.walls:
            pygame.draw.rect(self.game_display, self.BLUE, bbox.rect(int(self.DISPLAY_WIDTH/2), int(self.DISPLAY_HEIGHT/2), self.SCALE))

        xmin = -self.dim*self.SCALE + self.DISPLAY_WIDTH/2
        xmax = self.dim*self.SCALE + self.DISPLAY_WIDTH/2
        ymin = -self.dim*self.SCALE + self.DISPLAY_HEIGHT/2
        ymax = self.dim*self.SCALE + self.DISPLAY_HEIGHT/2

        pygame.draw.line(self.game_display, self.BLUE, (xmin, ymin), (xmin, ymax))
        pygame.draw.line(self.game_display, self.BLUE, (xmin, ymax), (xmax, ymax))
        pygame.draw.line(self.game_display, self.BLUE, (xmax, ymin), (xmax, ymax))
        pygame.draw.line(self.game_display, self.BLUE, (xmin, ymin), (xmax, ymin))

        # Draw agent
        # Take agent (x,y), change y sign, scale and translate
        agent_x, agent_y = self.state * np.array([1, -1], dtype=np.int32) * self.SCALE + np.array([self.DISPLAY_WIDTH/2, self.DISPLAY_HEIGHT/2], dtype=np.float32)
        pygame.draw.circle(self.game_display, self.RED, (int(agent_x), int(agent_y)), self.AGENT_RADIUS)

        # Update screen
        pygame.display.update()

    def step(self, action):
        assert action.shape == self.action_space.shape

        x, y = self.state

        dx = action[0]
        dx = np.clip(dx, -self.max_delta, self.max_delta)

        dy = action[1]
        dy = np.clip(dy, -self.max_delta, self.max_delta)

        new_x = x + dx
        new_y = y + dy

        # Check hit with a wall
        for bbox in self.walls:
            if bbox.check_if_inside(new_x, new_y):
                new_x, new_y = x, y

        if np.abs(new_x) >= self.dim or np.abs(new_y) >= self.dim:
            new_x, new_y = x, y

        self.state = np.array([new_x, new_y], dtype=np.float32)
        done = False
        reward = 0

        return self.state, reward, done, {}
