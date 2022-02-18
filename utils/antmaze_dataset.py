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

#from d4rl.offline_env import get_keys
import h5py
import numpy as np
import os
import torch
from tqdm import tqdm


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='medium', help='Maze type. umaze, medium, or large')
    parser.add_argument('--num_samples', type=int, default=int(1e6), help='Num samples to collect')
    parser.add_argument('--policy', type=str, default=None, help='Path to policy')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--max_episode_steps', type=int, default=1000)
    parser.add_argument('--no_multi_start', action='store_false', dest='multi_start')
    parser.add_argument('--no_multigoal', action='store_false', dest='multigoal')
    parser.add_argument('--no_noisy', action='store_false', dest='noisy')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--expert_demos', type=int, default=0, help='Number of expert demos to collect')
    parser.add_argument('--seed', type=int, default=0)
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
        if k == 'seed':
            dtype = int
        elif k in ['terminals', 'timeouts']:
            dtype = bool
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


def load_policy(policy_path):
    # temporarily set import path to avoid import error while loading
    import importlib, sys
    sys.path.append(os.path.dirname(importlib.util.find_spec('d4rl').origin))
    data = torch.load(policy_path)
    del sys.path[-1]
    policy = data['exploration/policy'].to('cpu')
    policy.eval()
    env = data['evaluation/env']
    return policy, env


def save_video(save_dir, file_name, frames, episode_id=0):
    from PIL import Image
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
    from utils.env_utils import make_antmaze_train_env, make_env, SilentEnv
    if verbose:
        print('Start collecting dataset')
        if args.expert_demos:
            print('Colleting {} expert demo data'.format(args.expert_demos))

    args.multigoal = True  # not used, always set as True

    if args.expert_demos:
        env = make_env('antmaze', args.task, seed=args.seed)
    else:
        env = make_antmaze_train_env(maze=args.task, sparse_reward=False, multi_start=args.multi_start)
        env = SilentEnv(env)
        env.seed(args.seed)
        env.observation_space.seed(args.seed)
        env.action_space.seed(args.seed)

    # Load the policy
    policy, _ = load_policy(args.policy)
    if verbose:
        print("Policy loaded from {}".format(args.policy))

    # Define goal reaching policy fn
    def _goal_reaching_policy_fn(obs, goal):
        goal_x, goal_y = goal
        obs_new = obs[2:]
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

    obs, ret, done = env.reset(), 0., False
    ret_sparse = 0.
    ts, timeout = 0, False

    if args.video:
        frames = [env.physics.render(width=500, height=500, depth=False)]

    episode_idx = 0

    for i in range(args.num_samples):
        with torch.no_grad():
            act, _ = data_collection_policy(obs)

        if args.noisy:
            act = act + np.random.randn(*act.shape)*0.2
            act = np.clip(act, -1.0, 1.0)

        next_obs, r, done, _ = env.step(act)
        ret += r

        # sparse reward calculation follows the GoalReachingEnv implementation in D4RL
        # https://github.com/rail-berkeley/d4rl/blob/7ab5a05c8782099e748ab30ca921021644478a62/d4rl/locomotion/goal_reaching_env.py
        if args.expert_demos:
            r_sparse = r
        else:
            r_sparse = 1.0 if r >= -0.5 else 0.0
        ret_sparse += r_sparse

        ts += 1

        if not done and (ts >= args.max_episode_steps or i == args.num_samples - 1):
            timeout = True
            done = True

        append_data(episode_data, obs, act, r, env.target_goal, done, timeout, env.physics.data, r_sparse)

        if args.video:
            frames.append(env.physics.render(width=500, height=500, depth=False))

        if done:
            episode_data['next_observations'] = episode_data['observations'][1:] + [next_obs]
            if not args.expert_demos or ret_sparse > 0:  # for expert demos, collect only the successful episodes
                merge_data(full_data, episode_data)
                returns.append(ret)
                returns_sparse.append(ret_sparse)

                episode_idx += 1
                if verbose and episode_idx % 10 == 0:
                    print('episode: {}, current size: {}, ret: {}, success: {}'.format(
                        episode_idx, len(full_data['observations']), ret, ret_sparse))

                if args.video:
                    frames = np.array(frames)
                    save_video('./videos/', 'Ant_navigation', frames, len(returns))

            episode_data = reset_data()
            obs, ret, done = env.reset(), 0., False
            ret_sparse = 0.
            ts, timeout = 0, False

            if args.video:
                frames = [env.physics.render(width=500, height=500, depth=False)]
        else:
            obs = next_obs

        if args.expert_demos and episode_idx >= args.expert_demos:
            break

    if args.expert_demos:
        assert episode_idx == args.expert_demos
    else:
        assert len(full_data['observations']) == len(full_data['actions']) == args.num_samples
    full_data['seed'] = [args.seed]
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


def get_keys(h5file):
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def load_hdf5(filepath):
    dataset = {}
    try:
        with h5py.File(filepath, 'r') as f:
            for k in tqdm(get_keys(f), desc="load datafile"):
                try:  # first try loading as an array
                    dataset[k] = f[k][:]
                except ValueError as e:  # try loading as a scalar
                    dataset[k] = f[k][()]
        return dataset
    except (FileNotFoundError, OSError) as e:
        return {}


def get_dataset(args, verbose=False, create_on_fail=False):
    args.expert_demos = args.expert_demos if hasattr(args, 'expert_demos') else 0
    if args.expert_demos:
        fn = 'Ant_maze_{}_expert_{}.hdf5'.format(args.task, args.expert_demos)
    else:
        fn = 'Ant_maze_{}{}_multistart_{}_multigoal_{}.hdf5'.format(
            args.task, '_noisy' if args.noisy else '', args.multi_start, args.multigoal)

    filepath = os.path.join(args.data_dir, fn)
    if verbose:
        print('Dataset file path: {}'.format(filepath))

    dataset = load_hdf5(filepath)
    if dataset:
        return dataset

    if not create_on_fail:
        raise FileNotFoundError('dataset file {} does not exist'.format(filepath))

    print('Dataset file not found. Trying to generate a new one.')

    if args.policy is None:
        args.policy = os.path.join(args.data_dir, 'antmaze_policy.pkl')
    if not os.path.exists(args.policy):
        download_policy_file(args.policy)

    return collect_dataset(args, filepath, verbose)


if __name__ == '__main__':
    from utils.python_utils import seed_all
    args = parse_args()
    print('seed is: {}'.format(args.seed))
    seed_all(args.seed)
    dataset = get_dataset(args, verbose=True, create_on_fail=True)
    print('dataset loaded. seed={}, obs shape={}'.format(
        dataset['seed'][0], dataset['observations'].shape))
