import torch
import gym
import numpy as np
import os
import datetime
import time

from tabulate import tabulate
from torch.utils import tensorboard
from joblib import Parallel, delayed
from src.utils.dtypes import float_type, int_type


def assign_flat_params(model, flat_params):
    """
    Assigns the flattened parameter vector to the model.
    """
    sections = [torch.reshape(w, (-1,)).shape[0] for w in model.parameters()]
    params = torch.split(flat_params, sections, dim=0)

    for i, p in enumerate(model.parameters()):
        p.data = torch.reshape(params[i], p.shape)


def backtracking(model, compute_objective, compute_constraint, constrain_thresh,
                 search_dir, step, max_iters=10, just_check_constraint=False):
    old_objective = compute_objective()
    old_params = torch.cat(
        [torch.reshape(w, (-1,)) for w in model.parameters()], dim=0
    )

    for i in range(max_iters):
        alpha = 0.5**i
        new_params = old_params + alpha * step * search_dir

        # Assign parameters after applying proposed step
        assign_flat_params(model, new_params)
        new_objective = compute_objective()

        new_constrain = compute_constraint()

        improvement = new_objective - old_objective

        # If we improve the constrained objective
        valid_update = (just_check_constraint or
                        (torch.isfinite(new_objective) and improvement > 0))

        # If the constraint is satisfied
        valid_update = (valid_update
                        and torch.isfinite(new_constrain)
                        and new_constrain < constrain_thresh)

        if valid_update:
            return True, new_params, i

    # We have not found a suitable step in max_iters
    assign_flat_params(model, old_params)
    return False, old_params, i


def conj_gradient(Ax, b, iters):
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()

    i = 0

    while True:
        Ap = Ax(p)

        rr = torch.dot(r, r)
        alpha = rr / torch.dot(p, Ap)
        x += alpha * p

        i += 1
        if i == iters:
            break

        r_new = r - alpha * Ap
        beta = torch.dot(r_new, r_new) / rr
        p = r_new + beta * p
        r = r_new

    return x

def collect_sample_batch(env, policy, batch_size, traj_len, num_workers=1):
    def _collect_sample_batch(env, policy, batch_size,
                              traj_len, parallel=False):
        steps = 0
        obs_shape = env.num_features

        total_states = None
        total_actions = None
        total_rewards = None
        total_real_traj_lens = None
        dones = None

        while steps < batch_size:

            if parallel:
                env.seed(np.random.randint(2**16))

            states = np.zeros(
                (1, traj_len + 1, obs_shape), dtype=np.float32
            )
            if type(env.action_space) == gym.spaces.Discrete:
                act_shape = 1
                actions = np.zeros((1, traj_len, act_shape), dtype=np.int32)
            else:
                act_shape = env.action_space.shape[0]
                actions = np.zeros(
                    (1, traj_len, act_shape), dtype=np.float32
                )
            rewards = np.zeros((1, traj_len), dtype=np.float32)
            real_traj_lens = np.zeros((1, 1), dtype=np.int32)
            dones = np.zeros((1, 1), dtype=np.bool)

            s = env.reset()

            for t in range(traj_len):

                states[0, t] = s

                a = policy.predict(s).numpy()

                actions[0, t] = a

                ns, r, done, _ = env.step(a)

                rewards[0, t] = r
                s = ns

                steps += 1

                if done is True or steps == batch_size:
                    break

            real_traj_lens[0, 0] = t+1
            dones[0, 0] = done
            states[0, t+1] = s

            if total_states is None:
                total_states = states
                total_actions = actions
                total_rewards = rewards
                total_real_traj_lens = real_traj_lens
                total_dones = dones
            else:
                total_states = np.vstack([total_states, states])
                total_actions = np.vstack([total_actions, actions])
                total_rewards = np.vstack([total_rewards, rewards])
                total_real_traj_lens = np.vstack([total_real_traj_lens,
                                                  real_traj_lens])
                total_dones = np.vstack([total_dones, dones])

        return (total_states, total_actions, total_rewards,
                total_real_traj_lens, total_dones)

    if num_workers == 1:
        return _collect_sample_batch(env, policy, batch_size, traj_len)
    else:
        assert batch_size % num_workers == 0, \
            "Please provide a batch size" \
            "that is a multiple of the worker size"

        batch_per_worker = int(batch_size / num_workers)
        results = Parallel(n_jobs=num_workers)(
            delayed(_collect_sample_batch)(
                env, policy, batch_per_worker, traj_len, parallel=True
            ) for i in range(num_workers)
        )
        return [np.vstack(x) for x in zip(*results)]


def process_traj(gamma, lambd, vfuncs, states, actions, rewards, boot_value):
    """
    Process a trajectory to compute
        - the targets (discounted sum of rewards)
        - the advantages
            (using generalized advantage estimation
             https://arxiv.org/abs/1506.02438)
    """
    real_traj_len = states.shape[0]

    # Compute targets
    targets = np.zeros((real_traj_len), dtype=np.float32)
    curr_target = boot_value
    for i in reversed(range(real_traj_len)):
        targets[i] = rewards[i] + gamma * curr_target
        curr_target = targets[i]

    # Compute advantages
    advantages = np.zeros((real_traj_len), dtype=np.float32)
    curr_advantage = 0
    for i in reversed(range(real_traj_len)):
        v_next = boot_value if i == real_traj_len - 1 else vfuncs[i+1]
        advantages[i] = ((rewards[i] + gamma * v_next - vfuncs[i])
                         + gamma * lambd * curr_advantage)
        curr_advantage = advantages[i]

    return targets, advantages


def trpo(
    env_maker,
    env_name,
    num_epochs,
    batch_size,
    traj_len,
    gamma=0.995,
    lambd=0.98,
    vfunc=None,
    policy=None,
    optimizer='lbfgs',
    critic_lr=1e-2,
    critic_reg=1e-3,
    critic_iters=1,
    critic_batch_size=64,
    cg_iters=10,
    cg_damping=0.1,
    kl_thresh=0.01,
    num_workers=1,
    out_path=None,
    seed=None
):
    # Create environment
    env = env_maker()

    # Define action type
    if type(env.action_space) == gym.spaces.Discrete:
        env_action_type = 'discrete'
    else:
        env_action_type = 'continuous'

    # Seed everything
    if seed is None:
        seed = np.random.randint(2**16)
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Set value function optimizer
    if optimizer == 'adam':
        vfunc_optimizer = torch.optim.Adam(vfunc.parameters(), lr=critic_lr)
    elif optimizer == 'lbfgs':
        vfunc_optimizer = torch.optim.LBFGS(
            vfunc.parameters(),
            lr=critic_lr,
            max_iter=25
        )
    else:
        raise NotImplementedError()

    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")
    csv_file_1 = open(os.path.join(out_path, f"{env_name}.csv"), 'w')
    csv_file_1.write(",".join(['Epoch', 'NumSamples', 'ExecutionTime', 'AverageReturn',
                               'BacktrackSuccess','BacktrackIters']))
    csv_file_1.write("\n")
    csv_file_1.flush()

    # Train loop
    num_samples = 0
    for epoch in range(num_epochs):

        t0 = time.time()

        # Collect trajectories
        states, actions, rewards, real_traj_lens, dones = collect_sample_batch(
            env, policy, batch_size, traj_len, num_workers
        )
        num_traj = states.shape[0]

        total_reward = 0

        # Process each trajectory to get the targets and advantages
        # needed for the update
        for traj in range(num_traj):
            real_traj_len = real_traj_lens[traj, 0]

            traj_states = states[traj, :real_traj_len]
            traj_actions = actions[traj, :real_traj_len]
            traj_rewards = rewards[traj, :real_traj_len]
            traj_vfuncs = vfunc(torch.from_numpy(traj_states).type(float_type))
            traj_done = dones[traj, 0]

            # Last state value is null because of termination
            # or bootstrapped because of maximum taken steps
            boot_value = (
                vfunc(torch.from_numpy(states[traj, -1, :]).type(float_type))
                if not traj_done
                else 0
            )

            # Get the targets and the advantages for each trajectory
            traj_targets, traj_advantages = process_traj(
                gamma, lambd,
                traj_vfuncs,
                traj_states,
                traj_actions,
                traj_rewards,
                boot_value
            )

            # Incrementally build tensors for this epoch
            if traj == 0:
                epoch_states = traj_states
                epoch_actions = traj_actions
                epoch_targets = traj_targets
                epoch_advantages = traj_advantages
            else:
                epoch_states = np.concatenate(
                    [epoch_states, traj_states], axis=0
                )
                epoch_actions = np.concatenate(
                    [epoch_actions, traj_actions], axis=0
                )
                epoch_targets = np.concatenate(
                    [epoch_targets, traj_targets], axis=0
                )
                epoch_advantages = np.concatenate(
                    [epoch_advantages, traj_advantages], axis=0
                )

            total_reward += np.sum(traj_rewards)

        # Normalize advantages
        epoch_advantages = (epoch_advantages - epoch_advantages.mean()) / epoch_advantages.std()

        # Create torch tensors for downstream computation
        epoch_states = torch.from_numpy(epoch_states).type(float_type)
        epoch_actions = torch.from_numpy(epoch_actions).type(float_type)
        if env_action_type == 'discrete':
            epoch_actions = epoch_actions.long()
        else:
            epoch_actions = epoch_actions.type(float_type)
        epoch_advantages = torch.from_numpy(epoch_advantages).type(float_type)
        epoch_targets = torch.from_numpy(epoch_targets).unsqueeze(1).type(float_type)

        # TRPO optimization

        # Fixed log probs
        old_log_prob = policy.get_log_p(
            epoch_states, epoch_actions
        ).detach()

        def compute_gain():
            """
            Computes the gain of the new policy w.r.t the old one
            """
            new_log_prob = policy.get_log_p(
                epoch_states, epoch_actions
            )
            gain = torch.mean(
                torch.exp(new_log_prob - old_log_prob) * (epoch_advantages)
            )
            return gain

        if env_action_type == 'discrete':
            p0 = policy(epoch_states).detach()
        else:
            mu0, _ = policy(epoch_states)
            mu0 = mu0.detach()
            log_std0 = policy.log_std.detach()

        def compute_kl():
            """
            Computes KL(policy_old||policy_new)
            or according to the following notation KL(0||1)
            """
            if env_action_type == 'discrete':
                p1 = policy(epoch_states)
                return (p0*torch.log(p0/p1)).sum(dim=1).mean()
            else:
                mu1, _ = policy(epoch_states)
                log_std1 = policy.log_std

                var0 = torch.exp(log_std0)**2
                var1 = torch.exp(log_std1)**2
                return (
                    (0.5 * ((var0 + (mu1-mu0)**2) / (var1 + 1e-7) - 1)
                     + log_std1 - log_std0)
                ).sum(dim=1).mean()

        def hessian_vector_product(x):
            """
            Computes the product between the Hessian of the KL
            wrt the policy parameters and the tensor provided as input x
            """
            kl = compute_kl()
            grads = torch.autograd.grad(
                kl, policy.parameters(), create_graph=True
            )
            grads = torch.cat(
                [torch.reshape(grad, (-1,)) for grad in grads], dim=0
            )
            sum_kl_x = torch.sum(grads * x, dim=0)
            grads_2 = torch.autograd.grad(sum_kl_x, policy.parameters())
            grads_2 = torch.cat(
                [torch.reshape(grad, (-1,)) for grad in grads_2], dim=0
            )
            grads_2 += cg_damping * x
            return grads_2

        gain = compute_gain()
        g = torch.autograd.grad(gain, policy.parameters())
        g = torch.cat([torch.reshape(x, (-1,)) for x in g], dim=0)

        x = conj_gradient(hessian_vector_product, g, iters=cg_iters)

        # 1/lagrange_multiplier is the maximum step we can take
        # along the gradient direction
        lagrange_mult = torch.sqrt(
            torch.dot(x, hessian_vector_product(x)) / (2*kl_thresh)
        )

        # Backtracking to ensure improvement and *exact* KL constraint
        # after the update
        success, params, backtrack_iters = backtracking(
            policy, compute_gain, compute_kl, kl_thresh, x, 1/lagrange_mult
        )

        # Update the value critic
        if optimizer == 'lbfgs':
            def compute_vfunc_loss():
                vfunc_optimizer.zero_grad()
                state_values = vfunc(epoch_states)
                params = torch.cat(
                    [torch.reshape(w, (-1,)) for w in vfunc.parameters()],
                    dim=0
                )
                l2 = torch.sum(params**2, dim=0)
                loss = (torch.mean((state_values - epoch_targets)**2)
                        + critic_reg*l2)
                loss.backward()
                return loss
            loss = vfunc_optimizer.step(compute_vfunc_loss)
        else:
            dataset = torch.utils.data.TensorDataset(epoch_states,
                                                     epoch_targets)
            dloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=critic_batch_size,
                shuffle=True,
                drop_last=True
            )

            for _ in range(critic_iters):
                vfunc_optimizer.zero_grad()
                for mb_states, mb_targets in dloader:
                    vfunc_optimizer.zero_grad()
                    loss = torch.mean((vfunc(mb_states) - mb_targets)**2)
                    loss.backward()
                    vfunc_optimizer.step()

        num_samples += epoch_states.shape[0]
        execution_time = time.time() - t0

        average_return = total_reward / num_traj

        # Log statistics
        writer.add_scalar("Num samples", num_samples, global_step=epoch)
        writer.add_scalar("Execution time (s)", execution_time, global_step=epoch)
        writer.add_scalar("AverageReturn", average_return, global_step=epoch)
        writer.add_scalar("BacktrackSuccess", success, global_step=epoch)
        writer.add_scalar("BacktrackIters", backtrack_iters, global_step=epoch)

        table = []
        fancy_float = lambda f : f"{f:.3f}"
        table.extend([
            ["Epoch", epoch],
            ["Num samples", num_samples],
            ["Execution time (s)", fancy_float(execution_time)],
            ["AverageReturn", fancy_float(average_return)],
            ["BacktrackSuccess", success],
            ["BacktrackIters", backtrack_iters]
        ])
        fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')
        print(fancy_grid)
        log_file.write(fancy_grid)
        log_file.flush()

        csv_file_1.write(f"{epoch},{num_samples},{execution_time},{average_return},{success},{backtrack_iters}\n")
        csv_file_1.flush()

        # Save policy
        torch.save(
            policy.state_dict(),
            os.path.join(out_path, 'policy_weights')
        )
