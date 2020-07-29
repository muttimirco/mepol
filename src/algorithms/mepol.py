import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import torch
import scipy
import scipy.special
import time
import os

from tabulate import tabulate
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
from torch.utils import tensorboard

from src.utils.dtypes import float_type, int_type


def get_heatmap(env, policy, discretizer, num_episodes, num_steps,
                cmap, interp, labels):
    """
    Builds a log-probability state visitation heatmap by running
    the policy in env. The heatmap is built using the provided
    discretizer.
    """
    average_state_dist = discretizer.get_empty_mat()
    average_entropy = 0

    for _ in range(num_episodes):
        s = env.reset()
        state_dist = discretizer.get_empty_mat()

        for t in range(num_steps):
            a = policy.predict(s).numpy()

            s, _, done, _ = env.step(a)
            state_dist[discretizer.discretize(s)] += 1

            if done:
                break

        state_dist /= t+1
        average_state_dist += state_dist
        average_entropy += scipy.stats.entropy(state_dist.ravel())

    average_state_dist /= num_episodes
    average_entropy /= num_episodes

    plt.close()
    image_fig = plt.figure()

    plt.xticks([])
    plt.yticks([])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if len(average_state_dist.shape) == 2:
        log_p = np.ma.log(average_state_dist)
        log_p_ravel = log_p.ravel()
        min_log_p_ravel = np.min(log_p_ravel)
        second_min_log_p_ravel = np.min(log_p_ravel[log_p_ravel != min_log_p_ravel])
        log_p_ravel[np.argmin(log_p_ravel)] = second_min_log_p_ravel
        plt.imshow(log_p.filled(min_log_p_ravel), interpolation=interp, cmap=cmap)
    else:
        plt.bar([i for i in range(discretizer.bins_sizes[0])], average_state_dist)

    return average_state_dist, average_entropy, image_fig


def collect_particles(env, policy, num_traj, traj_len, state_filter):
    """
    Collects num_traj * traj_len samples by running policy in the env.
    """
    states = np.zeros((num_traj, traj_len + 1, env.num_features), dtype=np.float32)
    actions = np.zeros((num_traj, traj_len, env.action_space.shape[0]), dtype=np.float32)
    real_traj_lengths = np.zeros((num_traj, 1), dtype=np.int32)

    for trajectory in range(num_traj):
        s = env.reset()

        for t in range(traj_len):
            states[trajectory, t] = s

            a = policy.predict(s).numpy()

            actions[trajectory, t] = a

            ns, _, done, _ = env.step(a)

            s = ns

            if done:
                break

        states[trajectory, t+1] = s
        real_traj_lengths[trajectory] = t+1

    next_states = None
    for n_traj in range(num_traj):
        traj_real_len = real_traj_lengths[n_traj].item()
        traj_next_states = states[n_traj, 1:traj_real_len+1, :].reshape(-1, env.num_features)
        if next_states is None:
            next_states = traj_next_states
        else:
            next_states = np.concatenate([next_states, traj_next_states], axis=0)

    if state_filter is not None:
        # Filter particle features over which entropy is maximized
        next_states = next_states[:, state_filter]

    return states, actions, real_traj_lengths, next_states


def compute_importance_weights(behavioral_policy, target_policy, states, actions,
                               num_traj, real_traj_lengths):
    # Initialize to None for the first concat
    importance_weights = None

    # Compute the importance weights
    # build iw vector incrementally from trajectory particles
    for n_traj in range(num_traj):
        traj_length = real_traj_lengths[n_traj][0].item()

        traj_states = states[n_traj, :traj_length]
        traj_actions = actions[n_traj, :traj_length]

        traj_target_log_p = target_policy.get_log_p(traj_states, traj_actions)
        traj_behavior_log_p = behavioral_policy.get_log_p(traj_states, traj_actions)

        traj_particle_iw = torch.exp(torch.cumsum(traj_target_log_p - traj_behavior_log_p, dim=0))

        if importance_weights is None:
            importance_weights = traj_particle_iw
        else:
            importance_weights = torch.cat([importance_weights, traj_particle_iw], dim=0)

    # Normalize the weights
    importance_weights /= torch.sum(importance_weights)
    return importance_weights


def compute_entropy(behavioral_policy, target_policy, states, actions,
                    num_traj, real_traj_lengths, distances, indices, k, G, B, ns, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions,
                                                    num_traj, real_traj_lengths)
    # Compute objective function
    # compute weights sum for each particle
    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)
    # compute volume for each particle
    volumes = (torch.pow(distances[:, k], ns) * torch.pow(torch.tensor(np.pi), ns/2)) / G
    # compute entropy
    entropy = - torch.sum((weights_sum / k) * torch.log((weights_sum / (volumes + eps)) + eps)) + B

    return entropy


def compute_kl(behavioral_policy, target_policy, states, actions,
               num_traj, real_traj_lengths, distances, indices, k, eps):
    importance_weights = compute_importance_weights(behavioral_policy, target_policy, states, actions,
                                                    num_traj, real_traj_lengths)

    weights_sum = torch.sum(importance_weights[indices[:, :-1]], dim=1)

    # Compute KL divergence between behavioral and target policy
    N = importance_weights.shape[0]
    kl = (1 / N) * torch.sum(torch.log(k / (N * weights_sum) + eps))

    numeric_error = torch.isinf(kl) or torch.isnan(kl)

    # Minimum KL is zero
    # NOTE: do not remove epsilon factor
    kl = torch.max(torch.tensor(0.0), kl)

    return kl, numeric_error


def collect_particles_and_compute_knn(env, behavioral_policy, num_traj, traj_len,
                                      state_filter, k, num_workers):
    assert num_traj % num_workers == 0, "Please provide a number of trajectories " \
                                        "that can be equally split among workers"

    # Collect particles using behavioral policy
    res = Parallel(n_jobs=num_workers)(
        delayed(collect_particles)(env, behavioral_policy, int(num_traj/num_workers), traj_len, state_filter)
        for _ in range(num_workers)
    )
    states, actions, real_traj_lengths, next_states = [np.vstack(x) for x in zip(*res)]

    # Fit knn for the batch of collected particles
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto', n_jobs=num_workers)
    nbrs.fit(next_states)
    distances, indices = nbrs.kneighbors(next_states)

    # Return tensors so that downstream computations can be moved to any target device (#todo)
    states = torch.tensor(states, dtype=float_type)
    actions = torch.tensor(actions, dtype=float_type)
    next_states = torch.tensor(next_states, dtype=float_type)
    real_traj_lengths = torch.tensor(real_traj_lengths, dtype=int_type)
    distances = torch.tensor(distances, dtype=float_type)
    indices = torch.tensor(indices, dtype=int_type)

    return states, actions, real_traj_lengths, next_states, distances, indices


def log_epoch_statistics(writer, log_file, csv_file_1, csv_file_2, epoch,
                         loss, entropy, num_off_iters, execution_time, full_entropy,
                         heatmap_image, heatmap_entropy, backtrack_iters, backtrack_lr):
    # Log to Tensorboard
    writer.add_scalar("Loss", loss, global_step=epoch)
    writer.add_scalar("Entropy", entropy, global_step=epoch)
    writer.add_scalar("Execution time", execution_time, global_step=epoch)
    writer.add_scalar("Number off-policy iteration", num_off_iters, global_step=epoch)
    if full_entropy is not None:
        writer.add_scalar(f"Full Entropy:", full_entropy, global_step=epoch)

    if heatmap_image is not None:
        writer.add_figure(f"Heatmap", heatmap_image, global_step=epoch)
        writer.add_scalar(f"Discrete entropy", heatmap_entropy, global_step=epoch)

    # Prepare tabulate table
    table = []
    fancy_float = lambda f : f"{f:.3f}"
    table.extend([
        ["Epoch", epoch],
        ["Execution time (s)", fancy_float(execution_time)],
        ["Entropy", fancy_float(entropy)],
        ["Off-policy iters", num_off_iters]
    ])

    if heatmap_image is not None:
        table.extend([
            ["Heatmap entropy", fancy_float(heatmap_entropy)]
        ])

    if backtrack_iters is not None:
        table.extend([
            ["Backtrack iters", backtrack_iters],
        ])

    fancy_grid = tabulate(table, headers="firstrow", tablefmt="fancy_grid", numalign='right')

    # Log to csv file 1
    csv_file_1.write(f"{epoch},{loss},{entropy},{full_entropy},{num_off_iters},{execution_time}\n")
    csv_file_1.flush()

    # Log to csv file 2
    if heatmap_image is not None:
        csv_file_2.write(f"{epoch},{heatmap_entropy}\n")
        csv_file_2.flush()

    # Log to stdout and log file
    log_file.write(fancy_grid)
    log_file.flush()
    print(fancy_grid)


def log_off_iter_statistics(writer, csv_file_3, epoch, global_off_iter,
                            num_off_iter, entropy, kl, lr):
    # Log to csv file 3
    csv_file_3.write(f"{epoch},{num_off_iter},{entropy},{kl},{lr}\n")
    csv_file_3.flush()

    # Also log to tensorboard
    writer.add_scalar("Off policy iter Entropy", entropy, global_step=global_off_iter)
    writer.add_scalar("Off policy iter KL", kl, global_step=global_off_iter)


def policy_update(optimizer, behavioral_policy, target_policy, states, actions,
                  num_traj, traj_len, distances, indices, k, G, B, ns, eps):
    optimizer.zero_grad()

    # Maximize entropy
    loss = - compute_entropy(behavioral_policy, target_policy, states, actions,
                             num_traj, traj_len, distances, indices, k, G, B, ns, eps)

    numeric_error = torch.isinf(loss) or torch.isnan(loss)

    loss.backward()
    optimizer.step()

    return loss, numeric_error


def mepol(env, env_name, state_filter, create_policy, k, kl_threshold, max_off_iters,
          use_backtracking, backtrack_coeff, max_backtrack_try, eps,
          learning_rate, num_traj, traj_len, num_epochs, optimizer,
          full_entropy_traj_scale, full_entropy_k,
          heatmap_every, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
          heatmap_cmap, heatmap_labels, heatmap_interp,
          seed, out_path, num_workers):

    # Seed everything
    if seed is not None:
        # Seed everything
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)

    # Create a behavioral, a target policy and a tmp policy used to save valid target policies
    # (those with kl <= kl_threshold) during off policy opt
    behavioral_policy = create_policy(is_behavioral=True)
    target_policy = create_policy()
    last_valid_target_policy = create_policy()
    target_policy.load_state_dict(behavioral_policy.state_dict())
    last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())

    # Set optimizer
    if optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(target_policy.parameters(), lr=learning_rate)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(target_policy.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError

    # Create writer for tensorboard
    writer = tensorboard.SummaryWriter(out_path)

    # Create log files
    log_file = open(os.path.join((out_path), 'log_file.txt'), 'a', encoding="utf-8")

    csv_file_1 = open(os.path.join(out_path, f"{env_name}.csv"), 'w')
    csv_file_1.write(",".join(['epoch', 'loss', 'entropy', 'full_entropy', 'num_off_iters','execution_time']))
    csv_file_1.write("\n")

    if heatmap_discretizer is not None:
        csv_file_2 = open(os.path.join(out_path, f"{env_name}-heatmap.csv"), 'w')
        csv_file_2.write(",".join(['epoch', 'average_entropy']))
        csv_file_2.write("\n")
    else:
        csv_file_2 = None

    csv_file_3 = open(os.path.join(out_path, f"{env_name}_off_policy_iter.csv"), "w")
    csv_file_3.write(",".join(['epoch', 'off_policy_iter', 'entropy', 'kl', 'learning_rate']))
    csv_file_3.write("\n")

    # Fixed constants
    ns = len(state_filter) if (state_filter is not None) else env.num_features
    B = np.log(k) - scipy.special.digamma(k)
    full_B = np.log(full_entropy_k) - scipy.special.digamma(full_entropy_k)
    G = scipy.special.gamma(ns/2 + 1)

    # At epoch 0 do not optimize, just log stuff for the initial policy
    epoch = 0
    t0 = time.time()

    # Full entropy
    states, actions, real_traj_lengths, next_states, distances, indices = \
        collect_particles_and_compute_knn(env, behavioral_policy, num_traj * full_entropy_traj_scale,
                                          traj_len, state_filter, full_entropy_k, num_workers)

    with torch.no_grad():
        full_entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                       num_traj * full_entropy_traj_scale, real_traj_lengths,
                                       distances, indices, full_entropy_k, G, full_B, ns, eps)

    # Entropy
    states, actions, real_traj_lengths, next_states, distances, indices = \
        collect_particles_and_compute_knn(env, behavioral_policy, num_traj,
                                          traj_len, state_filter, k, num_workers)

    with torch.no_grad():
        entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                  num_traj, real_traj_lengths,
                                  distances, indices, k, G, B, ns, eps)

    execution_time = time.time() - t0
    full_entropy = full_entropy.numpy()
    entropy = entropy.numpy()
    loss = - entropy

    # Heatmap
    if heatmap_discretizer is not None:
        _, heatmap_entropy, heatmap_image = \
            get_heatmap(env, behavioral_policy, heatmap_discretizer, heatmap_episodes, heatmap_num_steps,
                        heatmap_cmap, heatmap_interp, heatmap_labels)
    else:
        heatmap_entropy = None
        heatmap_image = None

    # Save initial policy
    torch.save(behavioral_policy.state_dict(), os.path.join(out_path, f"{epoch}-policy"))

    # Log statistics for the initial policy
    log_epoch_statistics(
            writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
            epoch=epoch,
            loss=loss,
            entropy=entropy,
            execution_time=execution_time,
            num_off_iters=0,
            full_entropy=full_entropy,
            heatmap_image=heatmap_image,
            heatmap_entropy=heatmap_entropy,
            backtrack_iters=None,
            backtrack_lr=None
        )

    # Main Loop
    global_num_off_iters = 0

    if use_backtracking:
        original_lr = learning_rate

    while epoch < num_epochs:
        t0 = time.time()

        # Off policy optimization
        kl_threshold_reached = False
        last_valid_target_policy.load_state_dict(behavioral_policy.state_dict())
        num_off_iters = 0

        # Collect particles to optimize off policy
        states, actions, real_traj_lengths, next_states, distances, indices = \
                collect_particles_and_compute_knn(env, behavioral_policy, num_traj,
                                                  traj_len, state_filter, k, num_workers)

        if use_backtracking:
            learning_rate = original_lr

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            backtrack_iter = 1
        else:
            backtrack_iter = None

        while not kl_threshold_reached:
            # Optimize policy
            loss, numeric_error = policy_update(optimizer, behavioral_policy, target_policy, states,
                                                actions, num_traj, real_traj_lengths,
                                                distances, indices, k, G, B, ns, eps)
            entropy = - loss.detach().numpy()

            with torch.no_grad():
                kl, kl_numeric_error = compute_kl(behavioral_policy, target_policy, states,
                                                  actions, num_traj, real_traj_lengths,
                                                  distances, indices, k, eps)

            kl = kl.numpy()

            if not numeric_error and not kl_numeric_error and kl <= kl_threshold:
                # Valid update
                last_valid_target_policy.load_state_dict(target_policy.state_dict())
                num_off_iters += 1
                global_num_off_iters += 1

                # Log statistics for this off policy iteration
                log_off_iter_statistics(writer, csv_file_3, epoch, global_num_off_iters,
                                        num_off_iters - 1, entropy, kl, learning_rate)

            else:
                if use_backtracking:
                    # We are here either because we could not perform any update for this epoch
                    # or because we need to perform one last update
                    if not backtrack_iter == max_backtrack_try:
                        target_policy.load_state_dict(last_valid_target_policy.state_dict())

                        learning_rate = original_lr / (backtrack_coeff ** backtrack_iter)

                        for param_group in optimizer.param_groups:
                            param_group['lr'] = learning_rate

                        backtrack_iter += 1
                        continue

                # Do not accept the update, set exit condition to end the epoch
                kl_threshold_reached = True

            if use_backtracking and backtrack_iter > 1:
                # Just perform at most 1 step using backtracking
                kl_threshold_reached = True

            if num_off_iters == max_off_iters:
                # Set exit condition also if the maximum number
                # of off policy opt iterations has been reached
                kl_threshold_reached = True

            if kl_threshold_reached:
                # Compute entropy of new policy
                with torch.no_grad():
                    entropy = compute_entropy(last_valid_target_policy, last_valid_target_policy,
                                              states, actions, num_traj, real_traj_lengths,
                                              distances, indices, k, G, B, ns, eps)

                if torch.isnan(entropy) or torch.isinf(entropy):
                    print("Aborting because final entropy is nan or inf...")
                    print("There is most likely a problem in knn aliasing. Use a higher k.")
                    exit()
                else:
                    # End of epoch, prepare statistics to log
                    epoch += 1

                    # Update behavioral policy
                    behavioral_policy.load_state_dict(last_valid_target_policy.state_dict())
                    target_policy.load_state_dict(last_valid_target_policy.state_dict())

                    loss = - entropy.numpy()
                    entropy = entropy.numpy()
                    execution_time = time.time() - t0

                    if epoch % heatmap_every == 0:
                        # Heatmap
                        if heatmap_discretizer is not None:
                            _, heatmap_entropy, heatmap_image = \
                                get_heatmap(env, behavioral_policy, heatmap_discretizer, heatmap_episodes,
                                            heatmap_num_steps, heatmap_cmap, heatmap_interp, heatmap_labels)
                        else:
                            heatmap_entropy = None
                            heatmap_image = None

                        # Full entropy
                        states, actions, real_traj_lengths, next_states, distances, indices = \
                            collect_particles_and_compute_knn(env, behavioral_policy, num_traj * full_entropy_traj_scale,
                                                              traj_len, state_filter, full_entropy_k, num_workers)

                        with torch.no_grad():
                            full_entropy = compute_entropy(behavioral_policy, behavioral_policy, states, actions,
                                                           num_traj * full_entropy_traj_scale, real_traj_lengths,
                                                           distances, indices, full_entropy_k, G, full_B, ns, eps)

                        full_entropy = full_entropy.numpy()

                        # Save policy
                        torch.save(behavioral_policy.state_dict(), os.path.join(out_path, f"{epoch}-policy"))

                    else:
                        heatmap_entropy = None
                        heatmap_image = None

                    # Log statistics for this epoch
                    log_epoch_statistics(
                        writer=writer, log_file=log_file, csv_file_1=csv_file_1, csv_file_2=csv_file_2,
                        epoch=epoch,
                        loss=loss,
                        entropy=entropy,
                        execution_time=execution_time,
                        num_off_iters=num_off_iters,
                        full_entropy=full_entropy,
                        heatmap_image=heatmap_image,
                        heatmap_entropy=heatmap_entropy,
                        backtrack_iters=backtrack_iter,
                        backtrack_lr=learning_rate
                    )

    return behavioral_policy