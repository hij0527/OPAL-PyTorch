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
    ap = argparse.ArgumentParser()

    ap.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    ap.add_argument('--gpu_id', type=str, default=None, help='GPU IDs to use. ex: 1,2 (default: None)')
    ap.add_argument('--verbose', action='store_true', help='set verbose mode')

    ap.add_argument('--opal_ckpt', type=str, required=True, help='path to OPAL checkpoint')
    ap.add_argument('--task_ckpt', type=str, required=True, help='path to task policy checkpoint')
    ap.add_argument('--test_episodes', type=int, default=10, help='number of episodes to test')

    ap.add_argument('--domain_name', type=str, choices=env_utils.DOMAIN_NAMES, default=env_utils.DOMAIN_NAMES[0], help='environment domain name')
    ap.add_argument('--task_name', type=str, choices=env_utils.TASK_NAMES, default=env_utils.TASK_NAMES[0], help='environment task name')
    ap.add_argument('--sparse_reward', action='store_true', help='sparse reward mode')
    ap.add_argument('--subtraj_len', '-C', type=int, default=10, help='length of subtrajectory (c)')
    ap.add_argument('--latent_dim', '-Z', type=int, default=8, help='dimension of primitive latent vector (dim(Z))')

    # model parameters
    ap.add_argument('--opal_hidden_size', '-H', metavar='H', type=int, default=200, help='OPAL: size of hidden layers (H)')
    ap.add_argument('--opal_num_layers', type=int, default=2, help='OPAL: number of hidden layers')
    ap.add_argument('--opal_num_gru_layers', type=int, default=4, help='OPAL: number of GRU layers')
    ap.add_argument('--opal_state_agnostic', action='store_true', help='OPAL: if set, use state agnostic models')
    ap.add_argument('--opal_unit_prior_std', action='store_true', help='OPAL: if set, use fixed std=1 for prior')

    ap.add_argument('--hidden_size', type=int, default=256, help='size of hidden layers (H)')
    ap.add_argument('--num_layers', type=int, default=3, help='number of hidden layers')
    ap.add_argument('--num_gru_layers', type=int, default=4, help='number of GRU layers')

    # downstream task
    ap.add_argument('--task_type', type=str, choices=['offline', 'imitation', 'online', 'multitask'], default='offline', help='downstream task type')
    ap.add_argument('--policy_type', type=str, choices=['cql', 'bc', 'sac', 'ppo'], default='cql', help='RL algorithm to use for downstream learning')
    ap.add_argument('--batch_size', type=int, default=50, help='batch size')
    ap.add_argument('--num_workers', type=int, default=6, help='number of DataLoader workers')

    args = ap.parse_args()

    for attr in ['opal_ckpt', 'task_ckpt']:
        val = getattr(args, attr)
        setattr(args, attr, os.path.expanduser(val) if val else val)

    return args


def test(args):
    device = python_utils.get_device(args.gpu_id)
    python_utils.seed_all(args.seed)
    env = env_utils.get_env(args.domain_name, args.task_name, seed=args.seed)

    dim_s = env.observation_space.shape[0]
    dim_a = env.action_space.shape[0]
    dim_z = args.latent_dim

    # load trained OPAL
    opal = OPAL(
        dim_s=dim_s,
        dim_a=dim_a,
        dim_z=dim_z,
        device=device,
        hidden_size=args.opal_hidden_size,
        num_layers=args.opal_num_layers,
        num_gru_layers=args.opal_num_gru_layers,
        state_agnostic=args.opal_state_agnostic,
        unit_prior_std=args.opal_unit_prior_std,
    )
    opal.load_state_dict(torch.load(args.opal_ckpt))

    # load trained task model
    if args.task_type == 'offline':
        from models.tasks.offline_model import OfflineModel
        task_model = OfflineModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                  hidden_size=args.hidden_size, num_layers=args.num_layers)
    elif args.task_type == 'imitation':
        from models.tasks.imitation_model import ImitationModel
        task_model = ImitationModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                    hidden_size=args.hidden_size, num_layers=args.num_layers)
    elif args.task_type == 'online':
        from models.tasks.online_model import OnlineModel
        task_model = OnlineModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                 hidden_size=args.hidden_size, num_layers=args.num_layers)
    elif args.task_type == 'multitask':
        from models.tasks.multitask_model import MultitaskModel
        task_model = MultitaskModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                    hidden_size=args.hidden_size, num_layers=args.num_layers,
                                    update_epochs=args.multitask_updates_per_step)
    else:
        raise ValueError('Unknown task type: {}'.format(args.task_type))

    task_model.load_state_dict(torch.load(args.task_ckpt))

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
                action = task_model.get_action(obs, deterministic=True).cpu().numpy()
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
