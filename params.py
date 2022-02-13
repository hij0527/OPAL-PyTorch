import argparse
import os

import utils.env_utils as env_utils


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument('--run_tag', type=str, default='opal', help='tag to run id (default: opal)')
    ap.add_argument('--no_timetag', action='store_true', help='if set, do not append time string to run id')
    ap.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    ap.add_argument('--gpu_id', type=str, default=None, help='GPU IDs to use. ex: 1,2 (default: None)')
    ap.add_argument('--verbose', action='store_true', help='set verbose mode')

    ap.add_argument('--results_root', type=str, default='./results', help='root directory to logs, models, videos, etc.')
    ap.add_argument('--data_dir', type=str, default='./data', help='dataset directory')
    ap.add_argument('--dataset_policy', type=str, default=None, help='path to policy file for dataset collection')

    ap.add_argument('--domain_name', type=str, choices=env_utils.DOMAIN_NAMES, default=env_utils.DOMAIN_NAMES[0], help='environment domain name')
    ap.add_argument('--task_name', type=str, choices=env_utils.TASK_NAMES, default=env_utils.TASK_NAMES[0], help='environment task name')
    ap.add_argument('--dataset_size', type=int, default=int(1e6), help='size of offline dataset for phase 1')
    ap.add_argument('--normalize', action='store_true', help='set to normalize states')
    ap.add_argument('--subtraj_len', '-C', metavar='c', type=int, default=10, help='length of subtrajectory (c)')
    ap.add_argument('--subtraj_num', '-N', metavar='N', type=int, default=-1, help='number of subtrajectories for phase 1 (N)')
    ap.add_argument('--sliding_window', action='store_true', help='if set, use sliding window for splitting subtrajectories')
    ap.add_argument('--latent_dim', '-Z', metavar='dim_Z', type=int, default=8, help='dimension of primitive latent vector (dim(Z))')

    # model parameters
    ap.add_argument('--hidden_size', '-H', metavar='H', type=int, default=200, help='size of hidden layers (H)')
    ap.add_argument('--num_layers', type=int, default=2, help='number of hidden layers')
    ap.add_argument('--num_gru_layers', type=int, default=4, help='number of GRU layers')
    ap.add_argument('--state_agnostic', action='store_true', help='if set, use state agnostic models')
    ap.add_argument('--unit_prior_std', action='store_true', help='if set, use fixed std=1 for prior')

    # phase 1: primitive training
    ap.add_argument('--epochs', type=int, default=100, help='number of epochs for phase 1')
    ap.add_argument('--batch_size', type=int, default=50, help='batch size for phase 1')
    ap.add_argument('--num_workers', type=int, default=6, help='number of DataLoader workers')
    ap.add_argument('--lr', type=float, default=1e-3, help='learning rate for phase 1')
    ap.add_argument('--truncate_normal', type=float, default=None, help='max sigma for sampling from normal')
    ap.add_argument('--eps_kld', type=float, default=0., help='upper bound for KL divergence contraint')
    ap.add_argument('--beta', type=float, default=0.1, help='weight of KL divergence in loss')
    ap.add_argument('--beta_final', type=float, default=None, help='if set, beta will linearly change to this value')
    ap.add_argument('--beta2', type=float, default=0., help='weight of additional regularization (default: 0)')
    ap.add_argument('--beta2_final', type=float, default=None, help='if set, beta2 will linearly change to this value')
    ap.add_argument('--grad_clip_val', type=float, default=0.001, help='gradient clipping value')
    ap.add_argument('--grad_clip_steps', type=int, default=100, help='steps to apply gradient clipping (set to -1 for always)')

    ap.add_argument('--print_freq', type=int, default=100, help='training log (stdout) frequency in steps')
    ap.add_argument('--log_freq', type=int, default=100, help='training log (tensorboard) frequency in steps')
    ap.add_argument('--save_freq', type=int, default=20, help='model save frequency in epochs')

    args = ap.parse_args()

    for attr in ['results_root', 'dataset_policy']:
        val = getattr(args, attr)
        setattr(args, attr, os.path.expanduser(val) if val else val)

    return args
