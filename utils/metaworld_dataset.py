import h5py
import metaworld
import numpy as np
import os
from tqdm import tqdm

from utils.env_utils import get_env_name


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='mt10', help='Task name. mt10 or mt50')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='Max episode step')
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
        else:
            dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)


_policy_class_names = {
    'peg-insert-side-v2': 'SawyerPegInsertionSideV2Policy',
}

def get_policy(env_name):
    import metaworld.policies
    policy_cls_name = 'Sawyer' + ''.join(s.capitalize() for s in env_name.split('-')) + 'Policy'
    if not hasattr(metaworld.policies, policy_cls_name):
        policy_cls_name = _policy_class_names[env_name]
    policy_cls = getattr(metaworld.policies, policy_cls_name)
    return policy_cls()


def collect_dataset(args, filepath, verbose):
    if args.task == 'mt10':
        benchmark_cls = metaworld.MT10
        num_repeats = 5
    elif args.task == 'mt50':
        benchmark_cls = metaworld.MT50
        num_repeats = 1
    else:
        raise ValueError

    full_data = []
    returns, returns_sparse = [], []

    for _ in range(num_repeats):
        benchmark = benchmark_cls()
        for task_idx, (env_name, env_cls) in enumerate(benchmark.train_classes.items()):
            env = env_cls()
            policy = get_policy(env_name)
            tasks = [task for task in benchmark.train_tasks if task.env_name == env_name]

            max_episode_steps = min(args.max_episode_steps, env.max_path_length)

            for task in tasks:
                env.set_task(task)
                obs, ret, done = env.reset(), 0., False
                ret_sparse = 0.
                step, timeout = 0, False

                while not done:
                    action = policy.get_action(obs)
                    next_obs, r, done, info = env.step(action)
                    assert info['success'] == 0 or info['success'] == 1
                    done |= bool(info['success'])
                    ret += r
                    step += 1

                    r_sparse = info['success']
                    ret_sparse += r_sparse

                    if not done and step >= max_episode_steps:
                        timeout = True
                        done = True

                    full_data.append([obs, action, r, next_obs, done, timeout, r_sparse, task_idx])

                    obs = next_obs

                returns.append(ret)
                returns_sparse.append(ret_sparse)

                if verbose:
                    print(' collecting... task: {}, current size: {}, ret: {:.2f}, success: {}'.format(
                        task_idx, len(full_data), ret, ret_sparse))

            env.close()

    data_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'timeouts', 'rewards_sparse', 'task_idx']
    full_data = {k: v for k, v in zip(data_keys, zip(*full_data))}
    npify(full_data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with h5py.File(filepath, 'w') as f:
        for k, v in full_data.items():
            f.create_dataset(k, data=v, compression='gzip')

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


def get_dataset(args, verbose=False):
    fn = 'Metaworld_{}.hdf5'.format(args.task.upper())
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

    return collect_dataset(args, filepath, verbose)


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args, verbose=True)
    print('dataset loaded. obs shape={}'.format(dataset['observations'].shape))
