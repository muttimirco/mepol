python src/experiments/goal_rl.py --env "GridGoal2" \
    --policy_init ./pretrained/grid_world --num_epochs 100 \
    --batch_size 24000 --traj_len 1200 --cg_iters 20 --kl_thresh 0.001