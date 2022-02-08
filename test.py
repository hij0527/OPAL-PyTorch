import argparse
import json
import numpy as np
import os
import time
import torch

from models.opal import OPAL
import utils.env_utils as env_utils
import utils.python_utils as python_utils


def parse_args():
    ap = argparse.ArgumentParser('OPAL training')

    ap.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    ap.add_argument('--gpu_id', type=str, default=None, help='GPU IDs to use. ex: 1,2 (default: None)')
    ap.add_argument('--verbose', action='store_true', help='set verbose mode')

    ap.add_argument('--ckpt_path', type=str, default=None, help='path to phase-2 model (checkpoint)')
    ap.add_argument('--test_episodes', type=int, default=10, help='number of episodes to test')

    ap.add_argument('--domain_name', type=str, choices=env_utils.DOMAIN_NAMES, default=env_utils.DOMAIN_NAMES[0], help='environment domain name')
    ap.add_argument('--task_name', type=str, choices=env_utils.TASK_NAMES, default=env_utils.TASK_NAMES[0], help='environment task name')
    ap.add_argument('--sparse_reward', action='store_true', help='sparse reward mode')
    ap.add_argument('--subtraj_len', '-C', type=int, default=10, help='length of subtrajectory (c)')
    ap.add_argument('--latent_dim', '-Z', type=int, default=8, help='dimension of primitive latent vector (dim(Z))')

    # model parameters
    ap.add_argument('--hidden_size', '-H', type=int, default=200, help='size of hidden layers (H)')
    ap.add_argument('--num_layers', type=int, default=2, help='number of hidden layers')
    ap.add_argument('--num_gru_layers', type=int, default=4, help='number of GRU layers')
    ap.add_argument('--task_hidden_size', type=int, default=256, help='size of hidden layers in task policy')
    ap.add_argument('--task_num_layers', type=int, default=2, help='number of hidden layers in task policy')

    args = ap.parse_args()

    return args


def test(args):
    device = python_utils.get_device(args.gpu_id)
    env = env_utils.get_env(args.domain_name, args.task_name)
    python_utils.seed_all(args.seed, env)

    dim_s = env.observation_space.shape[0]
    dim_a = env.action_space.shape[0]
    dim_z = args.latent_dim

    opal = OPAL(
        dim_s=dim_s,
        dim_a=dim_a,
        dim_z=dim_z,
        device=device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_gru_layers=args.num_gru_layers,
        task_hidden_size=args.task_hidden_size,
        task_num_layers=args.task_num_layers,
    )

    if args.ckpt_path is not None:
        opal.load_state_dict(torch.load(args.ckpt_path), phase=2)

    # rollout
    returns = []
    ep_lengths = []
    for ep_num in range(1, args.test_episodes + 1):
        tic = time.time()
        with python_utils.SuppressStdout():
            obs, ret, done = env.reset(), 0., False
        ep_len = 0

        while not done:
            with torch.no_grad():
                action = opal.get_action(obs, deterministic=True).cpu().numpy()
            with python_utils.SuppressStdout():
                next_obs, r, done, _ = env.step(action)
            ret += r
            ep_len += 1
            obs = next_obs

        returns.append(ret)
        ep_lengths.append(ep_len)
        print('[ep {}] ret: {:.2f}, len: {:d} (time: {:.2f}s)'.format(ep_num, ret, ep_len, time.time() - tic))

    print('avg return: {:.2f}'.format(np.mean(returns)))
    print('avg length: {:.1f}'.format(np.mean(ep_lengths)))


if __name__ == "__main__":
    args = parse_args()
    test(args)
