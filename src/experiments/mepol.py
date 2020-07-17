import argparse
import torch.nn as nn
import os
import numpy as np

from datetime import datetime
from src.envs.mountain_car_wall import MountainCarContinuous
from src.envs.gridworld_continuous import GridWorldContinuous
from src.envs.ant import Ant
from src.envs.hand_reach import HandReach
from src.envs.humanoid import Humanoid
from src.envs.discretizer import Discretizer
from src.envs.wrappers import ErgodicEnv
from src.algorithms.mepol import mepol
from src.policy import GaussianPolicy, train_supervised

parser = argparse.ArgumentParser(description='MEPOL')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting env trajectories and compute k-nn')
parser.add_argument('--env', type=str,
                    help='The MDP')
parser.add_argument('--zero_mean_start', type=int, default=1, choices=[0, 1],
                    help='Whether to make the policy start from a zero mean output')
parser.add_argument('--k', type=int, required=True,
                    help='The number of neighbors')
parser.add_argument('--kl_threshold', type=float, required=True,
                    help='The threshold after which the behavioral policy is updated')
parser.add_argument('--max_off_iters', type=int, default=20,
                    help='The maximum number of off policy optimization steps')
parser.add_argument('--use_backtracking', type=int, default=1, choices=[0, 1],
                    help='Whether to use backtracking or not')
parser.add_argument('--backtrack_coeff', type=float, default=2,
                    help='Backtrack coefficient')
parser.add_argument('--max_backtrack_try', type=int, default=10,
                    help='Maximum number of backtracking try')
parser.add_argument('--learning_rate', type=float, required=True,
                    help='The learning rate')
parser.add_argument('--num_trajectories', type=int, required=True,
                    help='The batch of trajectories used in off policy optimization')
parser.add_argument('--trajectory_length', type=int, required=True,
                    help='The maximum length of each trajectory in the batch of trajectories used in off policy optimization')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of epochs')
parser.add_argument('--optimizer', type=str, default='adam', choices=['rmsprop', 'adam'],
                    help='The optimizer')
parser.add_argument('--heatmap_every', type=int, default=10,
                    help='How many epochs to save a heatmap (if discretizer is defined).'
                         'Also the frequency at which policy weights are saved'
                         'Also the frequency at which full entropy is computed')
parser.add_argument('--heatmap_episodes', type=int, required=True,
                    help='The number of episodes on which the policy is run to compute the heatmap')
parser.add_argument('--heatmap_num_steps', type=int, required=True,
                    help='The number of steps per episode on which the policy is run to compute the heatmap')
parser.add_argument('--full_entropy_traj_scale', type=int, default=2,
                    help='The scale factor to be applied to the number of trajectories to compute the full entropy.')
parser.add_argument('--full_entropy_k', type=int, default=4,
                    help='The number of neighbors used to compute the full entropy')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='mepol',
                    help='The tensorboard directory under which the directory of this experiment is put')

args = parser.parse_args()

"""
Experiments specifications

    - env_create : callable that returns the target environment
    - discretizer_create : callable that returns a discretizer for the environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy
    - state_filter : list of indices representing the set of features over which entropy is maximized
    - eps : epsilon factor to deal with knn aliasing
    - heatmap_inter : kind of interpolation used to draw the heatmap
    - heatmap_cmap : heatmap color map
    - heatmap_labels : names of the discretized features

"""
exp_spec = {
    'MountainCar': {
        'env_create': lambda: ErgodicEnv(MountainCarContinuous()),
        'discretizer_create': lambda env: Discretizer([[env.min_position, env.max_position], [-env.max_speed, env.max_speed]], [12, 11]),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 1e-15,
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('Position', 'Velocity')
    },

    'GridWorld': {
        'env_create': lambda: ErgodicEnv(GridWorldContinuous()),
        'discretizer_create': lambda env: Discretizer([[-env.dim, env.dim], [-env.dim, env.dim]], [20, 20]),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
        'eps': 0,
        'heatmap_interp': None,
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('X', '-Y')
    },

    'Ant': {
        'env_create': lambda: ErgodicEnv(Ant()),
        'discretizer_create': lambda env: Discretizer([[-12.0, 12.0], [-12.0, 12.0]], [40, 40], lambda s: [s[0], s[1]]),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 0,
        'state_filter': list(range(7)),
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('X', 'Y')
    },

    # Higher-level
    'AntXY': {
        'env_create': lambda: ErgodicEnv(Ant()),
        'discretizer_create': lambda env: Discretizer([[-12.0, 12.0], [-12.0, 12.0]], [40, 40], lambda s: [s[0], s[1]]),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 0,
        'state_filter': list(range(2)),
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('X', 'Y')
    },

    'Humanoid': {
        'env_create': lambda: ErgodicEnv(Humanoid()),
        'discretizer_create': lambda env: Discretizer([[-12.0, 12.0], [-12.0, 12.0]], [40, 40], lambda s: [s[0], s[1]]),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 0,
        'state_filter': list(range(24)),
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('X', 'Y')
    },

    # Higher-level
    'HumanoidXYZ': {
        'env_create': lambda: ErgodicEnv(Humanoid()),
        'discretizer_create': lambda env: Discretizer([[-12.0, 12.0], [-12.0, 12.0]], [40, 40], lambda s: [s[0], s[1]]),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 0,
        'state_filter': list(range(3)),
        'heatmap_interp': 'spline16',
        'heatmap_cmap': 'Blues',
        'heatmap_labels': ('X', 'Y')
    },

    'HandReach': {
        'env_create': lambda: ErgodicEnv(HandReach()),
        'discretizer_create': lambda env: None,
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5,
        'eps': 0,
        'state_filter': list(range(24))
    },

}

spec = exp_spec.get(args.env)

if spec is None:
    print(f"Experiment name not found. Available ones are: {', '.join(key for key in exp_spec)}")

env = spec['env_create']()
discretizer = spec['discretizer_create'](env)
state_filter = spec.get('state_filter')
eps = spec['eps']

def create_policy(is_behavioral=False):

    policy = GaussianPolicy(
        num_features=env.num_features,
        hidden_sizes=spec['hidden_sizes'],
        action_dim=env.action_space.shape[0],
        activation=spec['activation'],
        log_std_init=spec['log_std_init']
    )

    if is_behavioral and args.zero_mean_start:
        policy = train_supervised(env, policy, train_steps=zero_mean_train_steps, batch_size=5000)

    return policy


exp_name = f"env={args.env},z_mu_start={args.zero_mean_start},k={args.k},kl_thresh={args.kl_threshold}," \
           f"max_off_iters={args.max_off_iters},num_traj={args.num_trajectories},traj_len={args.trajectory_length}," \
           f"lr={args.learning_rate},opt={args.optimizer},fe_traj_sc={args.full_entropy_traj_scale},fe_k={args.full_entropy_k}," \
           f"use_bt={args.use_backtracking},bt_coeff={args.backtrack_coeff},max_bt_try={args.max_backtrack_try}"

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results/exploration",
                        args.tb_dir_name, exp_name +
                        "__" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +
                        "__" + str(os.getpid()))
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write(f"Run info:\n")
    f.write("-"*10 + "\n")

    for key, value in vars(args).items():
        f.write(f"{key}={value}\n")

    f.write("-"*10 + "\n")

    tmp_policy = create_policy()
    f.write(tmp_policy.__str__())
    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2**16-1)
        f.write(f"Setting random seed {args.seed}\n")

mepol(
    env=env,
    env_name=args.env,
    state_filter=state_filter,
    create_policy=create_policy,
    k=args.k,
    kl_threshold=args.kl_threshold,
    max_off_iters=args.max_off_iters,
    use_backtracking=args.use_backtracking,
    backtrack_coeff=args.backtrack_coeff,
    max_backtrack_try=args.max_backtrack_try,
    eps=eps,
    learning_rate=args.learning_rate,
    num_traj=args.num_trajectories,
    traj_len=args.trajectory_length,
    num_epochs=args.num_epochs,
    optimizer=args.optimizer,
    full_entropy_traj_scale=args.full_entropy_traj_scale,
    full_entropy_k=args.full_entropy_k,
    heatmap_every=args.heatmap_every,
    heatmap_discretizer=discretizer,
    heatmap_episodes=args.heatmap_episodes,
    heatmap_num_steps=args.heatmap_num_steps,
    heatmap_cmap=spec.get('heatmap_cmap'),
    heatmap_labels=spec.get('heatmap_labels'),
    heatmap_interp=spec.get('heatmap_interp'),
    seed=args.seed,
    out_path=out_path,
    num_workers=args.num_workers
)