import argparse
import torch
import torch.nn as nn
import os
import numpy as np

from datetime import datetime
from src.envs.wrappers import CustomRewardEnv
from src.envs.gridworld_continuous import GridWorldContinuous
from src.envs.ant import Ant
from src.envs.upsidedown_ant import UpsideDownAnt
from src.envs.humanoid_standup import HumanoidStandup
from src.envs.humanoid import Humanoid
from src.algorithms.trpo import trpo
from src.policy import GaussianPolicy


parser = argparse.ArgumentParser(description='Goal-Based Reinforcement Learning - TRPO')

parser.add_argument('--num_workers', type=int, default=1,
                    help='How many parallel workers to use when collecting samples')
parser.add_argument('--env', type=str,
                    help='The MDP')
parser.add_argument('--policy_init', type=str, default=None,
                    help='Path to the weights for custom policy initialization.')
parser.add_argument('--num_epochs', type=int, required=True,
                    help='The number of training epochs')
parser.add_argument('--batch_size', type=int, required=True,
                    help='The batch size')
parser.add_argument('--traj_len', type=int, required=True,
                    help='The maximum length of a trajectory')
parser.add_argument('--gamma', type=float, default=0.995,
                    help='The discount factor')
parser.add_argument('--lambd', type=float, default=0.98,
                    help='The GAE lambda')
parser.add_argument('--optimizer', type=str, default='adam',
                    help='The optimizer used for the critic, either adam or lbfgs')
parser.add_argument('--critic_lr', type=float, default=1e-2,
                    help='Learning rate for critic optimization')
parser.add_argument('--critic_reg', type=float, default=1e-3,
                    help='Regularization coefficient for critic optimization')
parser.add_argument('--critic_iters', type=int, default=5,
                    help='Number of critic full updates')
parser.add_argument('--critic_batch_size', type=int, default=64,
                    help='Mini batch in case of adam optimizer for critic optimization')
parser.add_argument('--cg_iters', type=int, default=10,
                    help='Conjugate gradient iterations')
parser.add_argument('--cg_damping', type=float, default=0.1,
                    help='Conjugate gradient damping factor')
parser.add_argument('--kl_thresh', type=float, required=True,
                    help='KL threshold')
parser.add_argument('--seed', type=int, default=None,
                    help='The random seed')
parser.add_argument('--tb_dir_name', type=str, default='goal_rl',
                    help='The tensorboard directory under which the directory of this experiment is put')

args = parser.parse_args()

"""
Sparse reward functions
"""
def grid_goal1(s, r, d, i):
    if np.linalg.norm(s - np.array([5, 5], dtype=np.float32)) <= 1e-1:
        return 1, True
    else:
        return 0, False

def grid_goal2(s, r, d, i):
    if np.linalg.norm(s - np.array([2, 5], dtype=np.float32)) <= 1e-1:
        return 1, True
    else:
        return 0, False

def grid_goal3(s, r, d, i):
    if np.linalg.norm(s - np.array([5, 2], dtype=np.float32)) <= 1e-1:
        return 1, True
    else:
        return 0, False

def ant_escape(s, r, d, i):
    _self = i['self']
    l1 = _self.unwrapped.get_body_com('aux_1')[2]
    l2 = _self.unwrapped.get_body_com('aux_2')[2]
    l3 = _self.unwrapped.get_body_com('aux_3')[2]
    l4 = _self.unwrapped.get_body_com('aux_4')[2]
    thresh = 0.8
    if l1 >= thresh and l2 >= thresh and l3 >= thresh and l4 >= thresh:
        return 1, True
    else:
        return 0, False

def ant_navigate(s, r, d, i):
    if s[0] >= 7:
        return 1, True
    else:
        return 0, False

def ant_jump(s, r, d, i):
    if s[2] >= 3:
        return 1, True
    else:
        return 0, False

def humanoid_up(s, r, d, i):
    if s[2] >= 1:
        return 1, True
    else:
        return 0, False

"""
Experiments specifications

    - env_create : callable that returns the target environment
    - hidden_sizes : hidden layer sizes
    - activation : activation function used in the hidden layers
    - log_std_init : log_std initialization for GaussianPolicy

"""
exp_spec = {
    'GridGoal1': {
        'env_create': lambda: CustomRewardEnv(GridWorldContinuous(), grid_goal1),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
    },

    'GridGoal2': {
        'env_create': lambda: CustomRewardEnv(GridWorldContinuous(), grid_goal2),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
    },

    'GridGoal3': {
        'env_create': lambda: CustomRewardEnv(GridWorldContinuous(), grid_goal3),
        'hidden_sizes': [300, 300],
        'activation': nn.ReLU,
        'log_std_init': -1.5,
    },

    'AntEscape': {
        'env_create': lambda: CustomRewardEnv(UpsideDownAnt(), ant_escape),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5
    },

    'AntNavigate': {
        'env_create': lambda: CustomRewardEnv(Ant(), ant_navigate),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5
    },

    'AntJump': {
        'env_create': lambda: CustomRewardEnv(Ant(), ant_jump),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5
    },

    'HumanoidUp': {
        'env_create': lambda: CustomRewardEnv(HumanoidStandup(), humanoid_up),
        'hidden_sizes': [400, 300],
        'activation': nn.ReLU,
        'log_std_init': -0.5
    }

}

spec = exp_spec.get(args.env)

if spec is None:
    print("Experiment name not found. Available ones are: {}".format(', '.join(key for key in exp_spec)))

env = spec['env_create']()

# Create a policy
policy = GaussianPolicy(
    num_features=env.num_features,
    hidden_sizes=spec['hidden_sizes'],
    action_dim=env.action_space.shape[0],
    activation=spec['activation'],
    log_std_init=spec['log_std_init']
)

# Create a critic
hidden_sizes = [64, 64]
hidden_activation = nn.ReLU
layers = []
for i in range(len(hidden_sizes)):
    if i == 0:
        layers.extend([
            nn.Linear(env.num_features, hidden_sizes[i]),
            hidden_activation()
        ])
    else:
        layers.extend([
            nn.Linear(hidden_sizes[i-1], hidden_sizes[i]),
            hidden_activation()
        ])

layers.append(nn.Linear(hidden_sizes[i], 1))
vfunc = nn.Sequential(*layers)

for module in vfunc:
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight)


if args.policy_init is not None:
    kind = 'MEPOLInit'
    policy.load_state_dict(torch.load(args.policy_init))
else:
    kind = 'RandomInit'


exp_name = f"env={args.env},init={kind}"

out_path = os.path.join(os.path.dirname(__file__), "..", "..", "results/goal_rl",
                        args.tb_dir_name, exp_name +
                        "__" + datetime.now().strftime('%Y_%m_%d_%H_%M_%S') +
                        "__" + str(os.getpid()))
os.makedirs(out_path, exist_ok=True)

with open(os.path.join(out_path, 'log_info.txt'), 'w') as f:
    f.write("Run info:\n")
    f.write("-"*10 + "\n")

    for key, value in vars(args).items():
        f.write("{}={}\n".format(key, value))

    f.write("-"*10 + "\n")

    f.write(policy.__str__())
    f.write("-"*10 + "\n")
    f.write(vfunc.__str__())

    f.write("\n")

    if args.seed is None:
        args.seed = np.random.randint(2**16-1)
        f.write("Setting random seed {}\n".format(args.seed))

trpo(
    env_maker=spec['env_create'],
    env_name=args.env,
    num_epochs=args.num_epochs,
    batch_size=args.batch_size,
    traj_len=args.traj_len,
    gamma=args.gamma,
    lambd=args.lambd,
    vfunc=vfunc,
    policy=policy,
    optimizer=args.optimizer,
    critic_lr=args.critic_lr,
    critic_reg=args.critic_reg,
    critic_iters=args.critic_iters,
    critic_batch_size=args.critic_batch_size,
    cg_iters=args.cg_iters,
    cg_damping=args.cg_damping,
    kl_thresh=args.kl_thresh,
    num_workers=args.num_workers,
    out_path=out_path,
    seed=args.seed
)