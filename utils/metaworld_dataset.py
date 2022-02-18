import h5py
import metaworld
import numpy as np
import os
import random
from tqdm import tqdm


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='pick-place', help='Task name. mt10, mt50, or specific task')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--max_episode_steps', type=int, default=2000, help='Max episode step')
    parser.add_argument('--num_repeats', type=int, default=1000, help='Number of episodes per task')
    parser.add_argument('--no_noisy', action='store_false', dest='noisy')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def npify(data):
    for k in data:
        if k == 'seed':
            dtype = int
        elif k in ['terminals', 'timeouts']:
            dtype = bool
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
        env_name = None
    elif args.task == 'mt50':
        benchmark_cls = metaworld.MT50
        env_name = None
    else:  # specific task
        benchmark_cls = metaworld.MT1
        env_name = args.task
        if not env_name.endswith('-v2'):
            env_name += '-v2'

    full_data = []
    returns, returns_sparse = [], []

    for repeat_idx in range(args.num_repeats):
        seed = args.seed + repeat_idx * 10000
        if env_name is None:
            benchmark = benchmark_cls(seed=seed)
        else:
            benchmark = benchmark_cls(env_name, seed=seed)

        for task_idx, (env_name, env_cls) in enumerate(benchmark.train_classes.items()):
            env = env_cls()
            max_episode_steps = min(args.max_episode_steps, env.max_path_length)
            policy = get_policy(env_name)

            tasks = [task for task in benchmark.train_tasks if task.env_name == env_name]
            task = random.choice(tasks)
            env.set_task(task)

            obs, ret, done = env.reset(), 0., False
            ret_sparse = 0.
            step, timeout = 0, False

            while not done:
                action = policy.get_action(obs)
                if args.noisy:
                    action += np.random.randn(*action.shape) * 0.2
                    action = action.clip(env.action_space.low, env.action_space.high)

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
                print('iter: {} task: {}, current size: {}, ret: {:.2f}, success: {}'.format(
                    repeat_idx, task_idx, len(full_data), ret, ret_sparse))

            env.close()

    data_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'timeouts', 'rewards_sparse', 'task_idx']
    full_data = {k: v for k, v in zip(data_keys, zip(*full_data))}
    full_data['seed'] = [args.seed]
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


def get_dataset(args, verbose=False, create_on_fail=False):
    fn = 'Metaworld_{}{}.hdf5'.format(
        args.task, '_noisy' if args.noisy else '')
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
        if not create_on_fail:
            raise e

    return collect_dataset(args, filepath, verbose)


if __name__ == '__main__':
    from utils.python_utils import seed_all
    args = parse_args()
    seed_all(args.seed)
    dataset = get_dataset(args, verbose=True, create_on_fail=True)
    print('dataset loaded. seed={}, obs shape={}'.format(
        dataset['seed'][0], dataset['observations'].shape))
