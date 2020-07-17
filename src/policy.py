import numpy as np
import torch
import torch.nn as nn
import torch.functional as F

from src.utils.dtypes import float_type, int_type, eps

from collections import OrderedDict


class GaussianPolicy(nn.Module):
    """
    Gaussian Policy with state-independent diagonal covariance matrix
    """

    def __init__(self, hidden_sizes, num_features, action_dim, log_std_init=-0.5, activation=nn.ReLU):
        super().__init__()

        self.activation = activation

        layers = []
        layers.extend((nn.Linear(num_features, hidden_sizes[0]), self.activation()))
        for i in range(len(hidden_sizes) - 1):
            layers.extend((nn.Linear(hidden_sizes[i], hidden_sizes[i+1]), self.activation()))

        self.net = nn.Sequential(*layers)

        self.mean = nn.Linear(hidden_sizes[-1], action_dim)
        self.log_std = nn.Parameter(log_std_init * torch.ones(action_dim, dtype=float_type))

        # Constants
        self.register_buffer('log_of_pi', torch.tensor(np.log(np.pi), dtype=float_type))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.mean.weight)

        for l in self.net:
            if isinstance(l, nn.Linear):
                nn.init.xavier_uniform_(l.weight)

    def get_log_p(self, states, actions):
        mean, _ = self(states)
        return torch.sum(
            -0.5 * (
                self.log_of_pi
                + 2*self.log_std
                + ((actions - mean)**2 / (torch.exp(self.log_std) + eps)**2)
            ), dim=1
        )

    def forward(self, x, deterministic=False):
        mean = self.mean(self.net(x))

        if deterministic:
            output = mean
        else:
            output = mean + torch.randn(mean.size(), dtype=float_type) * torch.exp(self.log_std)

        return mean, output


    def predict(self, s, deterministic=False):
        with torch.no_grad():
            s = torch.tensor(s, dtype=float_type).unsqueeze(0)
            return self(s, deterministic=deterministic)[1][0]


def train_supervised(env, policy, train_steps=100, batch_size=5000):
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.00025)
    dict_like_obs = True if type(env.observation_space.sample()) is OrderedDict else False

    for _ in range(train_steps):
        optimizer.zero_grad()

        if dict_like_obs:
            states = torch.tensor([env.observation_space.sample()['observation'] for _ in range(5000)], dtype=float_type)
        else:
            states = torch.tensor([env.observation_space.sample()[:env.num_features] for _ in range(5000)], dtype=float_type)

        actions = policy(states)[0]
        loss = torch.mean((actions - torch.zeros_like(actions, dtype=float_type)) ** 2)

        loss.backward()
        optimizer.step()

    return policy