python src/experiments/goal_rl.py --env "AntNavigate" \
    --policy_init ./pretrained/higher_lvl_ant --num_epochs 1000 \
    --batch_size 20000 --traj_len 500 --cg_iters 20 --kl_thresh 0.01