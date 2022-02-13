import argparse
import os

import utils.env_utils as env_utils


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('--opal_ckpt', type=str, required=True, help='phase 1 checkpoint path')

    ap.add_argument('--run_tag', type=str, default=None, help='tag to run id (default: same as task_type)')
    ap.add_argument('--no_timetag', action='store_true', help='if set, do not append time string to run id')
    ap.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    ap.add_argument('--gpu_id', type=str, default=None, help='GPU IDs to use. ex: 1,2 (default: None)')
    ap.add_argument('--verbose', action='store_true', help='set verbose mode')

    ap.add_argument('--results_root', type=str, default='./results', help='root directory to logs, models, videos, etc.')
    ap.add_argument('--data_dir', type=str, default='./data', help='dataset directory')
    ap.add_argument('--dataset_policy', type=str, default=None, help='path to policy file for dataset collection')

    ap.add_argument('--domain_name', type=str, choices=env_utils.DOMAIN_NAMES, default=env_utils.DOMAIN_NAMES[0], help='environment domain name')
    ap.add_argument('--task_name', type=str, choices=env_utils.TASK_NAMES, default=env_utils.TASK_NAMES[0], help='environment task name')
    ap.add_argument('--sparse_reward', action='store_true', help='sparse reward mode')
    ap.add_argument('--dataset_size', type=int, default=int(1e6), help='size of offline dataset for phase 1')
    ap.add_argument('--normalize', action='store_true', help='set to normalize states')
    ap.add_argument('--subtraj_len', '-C', metavar='c', type=int, default=10, help='length of subtrajectory (c)')
    ap.add_argument('--subtraj_num', '-N', metavar='N', type=int, default=-1, help='number of subtrajectories for phase 1 (N)')
    ap.add_argument('--sliding_window', action='store_true', help='if set, use sliding window for splitting subtrajectories')
    ap.add_argument('--latent_dim', '-Z', metavar='dim_Z', type=int, default=8, help='dimension of primitive latent vector (dim(Z))')

    # model parameters
    ap.add_argument('--opal_hidden_size', '-H', metavar='H', type=int, default=200, help='OPAL: size of hidden layers (H)')
    ap.add_argument('--opal_num_layers', type=int, default=2, help='OPAL: number of hidden layers')
    ap.add_argument('--opal_num_gru_layers', type=int, default=4, help='OPAL: number of GRU layers')
    ap.add_argument('--opal_state_agnostic', action='store_true', help='OPAL: if set, use state agnostic models')
    ap.add_argument('--opal_unit_prior_std', action='store_true', help='OPAL: if set, use fixed std=1 for prior')

    ap.add_argument('--hidden_size', type=int, default=256, help='size of hidden layers (H)')
    ap.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
    ap.add_argument('--num_gru_layers', type=int, default=4, help='number of GRU layers')

    # data labeling
    ap.add_argument('--load_latent_buffer', type=str, default=None, help='if set, load latent buffer from this path')
    ap.add_argument('--save_latent_buffer', type=str, default=None, help='if set, save latent buffer to this path')

    # phase 2: downstream task training
    ap.add_argument('--task_type', type=str, choices=['offline', 'imitation', 'online', 'multitask'], default='offline', help='downstream task type')
    ap.add_argument('--policy_type', type=str, choices=['cql', 'bc', 'sac', 'ppo'], default='cql', help='RL algorithm to use for downstream learning')
    ap.add_argument('--batch_size', type=int, default=50, help='batch size for phase 2')
    ap.add_argument('--num_workers', type=int, default=6, help='number of DataLoader workers')
    ap.add_argument('--lr', type=float, default=3e-4, help='learning rate for phase 2')

    ap.add_argument('--print_freq', type=int, default=200, help='training log (stdout) frequency in steps')
    ap.add_argument('--log_freq', type=int, default=200, help='training log (tensorboard) frequency in steps')
    ap.add_argument('--save_freq', type=int, default=100, help='model save frequency in epochs/episodes')

    # offline
    ap.add_argument('--offline_finetune_epochs', type=int, default=30, help='')
    ap.add_argument('--offline_task_epochs', type=int, default=100, help='')

    # imitation
    ap.add_argument('--imitation_expert_policy', type=str, default=None, help='path to policy file for expert demo collection')
    ap.add_argument('--imitation_num_demos', type=int, default=10, help='')
    ap.add_argument('--imitation_finetune_epochs', type=int, default=50, help='')
    ap.add_argument('--imitation_task_epochs', type=int, default=100, help='')

    # online
    ap.add_argument('--online_train_steps', type=int, default=int(2.5e6), help='')
    ap.add_argument('--online_init_random_steps', type=int, default=10000, help='')
    ap.add_argument('--online_updates_per_step', type=int, default=1, help='')
    ap.add_argument('--online_batch_size', type=int, default=256, help='')
    ap.add_argument('--online_eval_freq', type=int, default=10, help='')
    ap.add_argument('--online_eval_episodes', type=int, default=5, help='')

    # multi-task
    ap.add_argument('--multitask_train_steps', type=int, default=int(1e7), help='')
    ap.add_argument('--multitask_update_interval', type=int, default=4000, help='')
    ap.add_argument('--multitask_updates_per_step', type=int, default=80, help='')
    ap.add_argument('--multitask_eval_freq', type=int, default=10, help='')
    ap.add_argument('--multitask_eval_episodes', type=int, default=5, help='')

    args = ap.parse_args()

    for attr in ['results_root', 'dataset_policy', 'opal_ckpt']:
        val = getattr(args, attr)
        setattr(args, attr, os.path.expanduser(val) if val else val)

    return args
