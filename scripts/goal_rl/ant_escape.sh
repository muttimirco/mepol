python src/experiments/goal_rl.py --env "AntEscape" \
    --policy_init ./pretrained/ant --num_epochs 500 \
    --batch_size 5000 --traj_len 500 --cg_iters 20 --kl_thresh 0.01