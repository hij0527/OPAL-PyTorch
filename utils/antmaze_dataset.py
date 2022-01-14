# Code adapted from official D4RL repository
# https://github.com/rail-berkeley/d4rl/blob/0ae0c1beeb948a62b5d8b04cec815d2250a75452/scripts/generation/generate_ant_maze_datasets.py
# https://github.com/rail-berkeley/d4rl/blob/0ae0c1beeb948a62b5d8b04cec815d2250a75452/scripts/generation/relabel_antmaze_rewards.py
# https://github.com/rail-berkeley/d4rl/blob/5aad1d72f470e9a4da64610a15a01f770be40998/d4rl/offline_env.py
#
# Notable changes
# - timestep (ts) is incremented before checking the timeout
# - timeouts are also marked as terminals
# - the last steps of each episode is marked as timeout and terminal if it was not terminal already
# - 'next_observations' and 'rewards_sparse' are added to the dataset
# - terminals are obtained manually, since the environment never returns True for terminal

import numpy as np
import h5py
import argparse
import d4rl
from d4rl.offline_env import get_keys
import torch
from tqdm import tqdm
from PIL import Image
import os

from utils.env_utils import get_antmaze_env
from utils.python_utils import SuppressStdout


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noisy', action='store_true', help='Noisy actions')
    parser.add_argument('--maze', type=str, default='umaze', help='Maze type. umaze, medium, or large')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--policy_file', type=str, default=None, help='Path to policy')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--max_episode_steps', default=1000, type=int)
    parser.add_argument('--multi_start', action='store_true')
    parser.add_argument('--multigoal', action='store_true')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save dataset')
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            'infos/qpos': [],
            'infos/qvel': [],
            'next_observations': [],
            'rewards_sparse': [],
            }


def append_data(data, s, a, r, tgt, done, timeout, env_data, r_sparse):
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['infos/goal'].append(tgt)
    data['infos/qpos'].append(env_data.qpos.ravel().copy())
    data['infos/qvel'].append(env_data.qvel.ravel().copy())
    data['rewards_sparse'].append(r_sparse)
    # next_observations is set separately


def merge_data(data1, data2):
    for k in data1.keys():
        data1[k] += data2[k]


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def load_policy(policy_file):
    # temporarily set import path to avoid import error while loading
    import sys
    sys.path.append(os.path.dirname(d4rl.__file__))
    data = torch.load(policy_file)
    policy = data['exploration/policy'].to('cpu')
    env = data['evaluation/env']
    return policy, env


def save_video(save_dir, file_name, frames, episode_id=0):
    filename = os.path.join(save_dir, file_name+ '_episode_{}'.format(episode_id))
    if not os.path.exists(filename):
        os.makedirs(filename)
    num_frames = frames.shape[0]
    for i in range(num_frames):
        img = Image.fromarray(np.flipud(frames[i]), 'RGB')
        img.save(os.path.join(filename, 'frame_{}.png'.format(i)))


def download_policy_file(policy_path):
    import requests
    policy_url = "http://rail.eecs.berkeley.edu/datasets/offline_rl/ant_hierarch_pol.pkl"
    print('Downloading policy from: {}'.format(policy_url))

    response = requests.get(policy_url, stream=True)
    length = int(response.headers.get('content-length', 0))

    os.makedirs(os.path.dirname(policy_path), exist_ok=True)

    with tqdm(total=length, unit='iB', unit_scale=True) as pbar, open(policy_path, 'wb') as f:
        for data in response.iter_content(1024):
            pbar.update(len(data))
            f.write(data)

    print('Policy downloaded to: {}'.format(policy_path))


def collect_dataset(args, filepath, verbose):
    if verbose:
        print('Start collecting dataset')

    args.multigoal = True  # not used, always set as True

    with SuppressStdout():
        env = get_antmaze_env(maze=args.maze, multi_start=args.multi_start)

    # Load the policy
    policy, _ = load_policy(args.policy_file)
    if verbose:
        print("Policy loaded from {}".format(args.policy_file))

    # Define goal reaching policy fn
    def _goal_reaching_policy_fn(obs, goal):
        goal_x, goal_y = goal
        obs_new = obs[2:-2]
        goal_tuple = np.array([goal_x, goal_y])

        # normalize the norm of the relative goals to in-distribution values
        goal_tuple = goal_tuple / np.linalg.norm(goal_tuple) * 10.0

        new_obs = np.concatenate([obs_new, goal_tuple], -1)
        return policy.get_action(new_obs)[0], (goal_tuple[0] + obs[0], goal_tuple[1] + obs[1])

    # create waypoint generating policy integrated with high level controller
    data_collection_policy = env.create_navigation_policy(_goal_reaching_policy_fn)

    full_data = reset_data()
    episode_data = reset_data()
    returns, returns_sparse = [], []

    with SuppressStdout():
        obs, ret, done = env.reset(), 0., False
        env.set_target()
    ret_sparse = 0.
    ts, timeout = 0, False

    if args.video:
        frames = [env.physics.render(width=500, height=500, depth=False)]

    for i in range(args.num_samples):
        with SuppressStdout():
            act, _ = data_collection_policy(obs)

        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.2
            act = np.clip(act, -1.0, 1.0)

        next_obs, r, done, _ = env.step(act)
        ret += r

        # Since eval=False for env, env will never emit done=True, and we have to set it manually.
        # done is the same as sparse reward, following the GoalReachingEnv implementation (L40, L42, L46)
        # https://github.com/rail-berkeley/d4rl/blob/7ab5a05c8782099e748ab30ca921021644478a62/d4rl/locomotion/goal_reaching_env.py
        r_sparse = 1.0 if r >= -0.5 else 0.0
        ret_sparse += r_sparse
        done = bool(r_sparse)

        ts += 1

        if not done and (ts >= args.max_episode_steps or i == args.num_samples - 1):
            timeout = True
            done = True

        append_data(episode_data, obs[:-2], act, r, env.target_goal, done, timeout, env.physics.data, r_sparse)

        if args.video:
            frames.append(env.physics.render(width=500, height=500, depth=False))

        if done:
            episode_data['next_observations'] = episode_data['observations'][1:] + [next_obs[:-2]]
            merge_data(full_data, episode_data)
            returns.append(ret)
            returns_sparse.append(ret_sparse)

            if args.video:
                frames = np.array(frames)
                save_video('./videos/', 'Ant_navigation', frames, len(returns))

            episode_data = reset_data()
            with SuppressStdout():
                obs, ret, done = env.reset(), 0., False
                env.set_target()
            ts, timeout = 0, False

            if args.video:
                frames = [env.physics.render(width=500, height=500, depth=False)]

            if verbose and len(full_data['observations']) % 10000 == 0:
                print(' collecting... (current size: {})'.format(len(full_data['observations'])))
        else:
            obs = next_obs

    assert(len(full_data['observations']) == len(full_data['actions']) == args.num_samples)

    npify(full_data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with h5py.File(filepath, 'w') as f:
        for k in full_data:
            f.create_dataset(k, data=full_data[k], compression='gzip')

    if verbose:
        print('Dataset saved: {}'.format(filepath))
        print('  obs shape: {}'.format(full_data['observations'].shape))
        print('  num episodes: {}'.format(len(returns)))
        print('  avg return: {:.2f}'.format(np.mean(returns)))
        print('  avg sparse return: {:.2f}'.format(np.mean(returns_sparse)))

    return full_data


def get_dataset(args, verbose=False):
    fn = 'Ant_maze_{}{}_multistart_{}_multigoal_{}.hdf5'.format(
        args.maze, '_noisy' if args.noisy else '', args.multi_start, args.multigoal)
    filepath = os.path.join(args.data_dir, fn)
    if verbose:
        print('Dataset file path: {}'.format(filepath))

    try:
        dataset = {}
        with h5py.File(filepath, 'r') as f:
            for k in tqdm(get_keys(f), desc="load datafile"):
                try:  # first try loading as an array
                    dataset[k] = f[k][:]
                except ValueError as e:  # try loading as a scalar
                    dataset[k] = f[k][()]
        return dataset
    except (FileNotFoundError, OSError) as e:
        print('Dataset file not found. Trying to generate a new one.')

    if args.policy_file is None:
        args.policy_file = os.path.join(args.data_dir, 'antmaze_policy.pkl')
    if not os.path.exists(args.policy_file):
        download_policy_file(args.policy_file)

    return collect_dataset(args, filepath, verbose)


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args, verbose=True)
