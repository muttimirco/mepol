python src/experiments/goal_rl.py --env "HumanoidUp" \
    --policy_init ./pretrained/higher_lvl_humanoid --num_epochs 1000 \
    --batch_size 20000 --traj_len 2000 --cg_iters 20 --kl_thresh 0.01