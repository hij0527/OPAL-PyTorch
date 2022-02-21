import gym
import d4rl
from d4rl.offline_env import set_dataset_path
import numpy as np
import os


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='partial', help='Task type. complete, partial, or mixed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save dataset')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def preprocess_data(dataset, env):
    # Due to REMOVE_TASKS_WHEN_COMPLETE=False during data collection,
    # the 'rewards' are cumulative rewards rather than sparse ones.
    # Recover the rewards by subtracting adjacent rewards.
    # Also, add sparse rewards (successes) and timeouts to the dataset.
    terminal_indices = np.where(dataset['terminals'])[0]
    terminal_indices = np.insert(terminal_indices, 0, -1)  # virtual terminal before the first transition
    if terminal_indices[-1] != len(dataset['terminals']) - 1:
        # treat the last transition as terminal
        terminal_indices = np.append(terminal_indices, len(dataset['terminals']) - 1)

    dataset['original_rewards'] = dataset['rewards'].copy()
    recovered_rewards = dataset['rewards'].copy()
    reward_for_success = env.env.ref_max_score
    successes = np.zeros(recovered_rewards.shape, dtype=bool)

    for i in range(1, len(terminal_indices)):
        start_idx, end_idx = terminal_indices[i - 1] + 1, terminal_indices[i]
        recovered_rewards[start_idx:end_idx+1] -= np.insert(dataset['rewards'][start_idx:end_idx], 0, 0)
        assert recovered_rewards[start_idx:end_idx+1].sum() == dataset['rewards'][end_idx]
        successes[end_idx] = dataset['rewards'][end_idx] == reward_for_success
        assert dataset['terminals'][end_idx] or not successes[end_idx]  # success should be terminal as well

    timeouts = np.logical_and(dataset['terminals'], np.logical_not(successes))

    dataset['rewards'] = recovered_rewards.copy()
    dataset['rewards_sparse'] = successes.copy()
    dataset['timeouts'] = timeouts.copy()

    return dataset


def get_dataset(args, verbose=False, create_on_fail=False):
    env = gym.make('kitchen-{}-v0'.format(args.task))
    set_dataset_path(args.data_dir)
    dataset = d4rl.qlearning_dataset(env)
    return preprocess_data(dataset, env)


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args)
    print('dataset loaded. size={}'.format(len(dataset['observations'])))
