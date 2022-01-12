from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset


class Buffer(Dataset):
    def __init__(self, domain_name, task_name, subtraj_len, verbose=False):
        self.domain_name = domain_name
        self.task_name = task_name
        self.subtraj_len = subtraj_len
        self.verbose = verbose

        self.dataset = {}
        self.terminal_indices = np.empty(0, dtype=int)
        self.subtraj_indices = np.empty(0, dtype=int)

    def __getitem__(self, index):
        # return subtrajectories
        subtraj_slice = np.arange(self.subtraj_len) + self.subtraj_indices[index]
        return {
            'observations': self.dataset['observations'][subtraj_slice],
            'actions': self.dataset['actions'][subtraj_slice],
            'rewards': self.dataset['rewards'][subtraj_slice],
            'next_observations': self.dataset['next_observations'][subtraj_slice],
            'terminals': self.dataset['terminals'][subtraj_slice],
        }

    def __len__(self):
        return len(self.subtraj_indices)

    def gather_data(self, sparse_reward=False, data_dir='./data', policy_path=None):
        # dataset: dict of np.ndarrays with keys including 'observations', 'actions', 'rewards', 'next_observations', 'terminals'
        if self.domain_name == 'antmaze':
            import argparse
            from utils.antmaze_dataset import get_dataset
            dataset = get_dataset(argparse.Namespace(
                noisy=True,
                maze=self.task_name,
                num_samples=int(1e6),
                policy_file=policy_path,
                video=False,
                max_episode_steps=1000,
                multi_start=True,
                multigoal=True,
                data_dir=data_dir,
                verbose=self.verbose,
            ))
        elif self.domain_name == 'kitchen':
            raise NotImplementedError
        elif self.domain_name == 'metaworld':
            raise NotImplementedError

        if sparse_reward:
            dataset['rewards'] = dataset['rewards_sparse']

        terminal_indices = np.where(dataset['terminals'])[0]
        terminal_indices = np.insert(terminal_indices, 0, -1)  # virtual terminal before the first transition
        if terminal_indices[-1] != len(dataset['terminals']) - 1:
            terminal_indices = np.append(terminal_indices, len(dataset['terminals']) - 1)  # treat the last transition as terminal

        self.dataset = dataset
        self.terminal_indices = terminal_indices

        if self.verbose:
            traj_lens = terminal_indices[1:] - terminal_indices[:-1]
            traj_rets = np.array([dataset['rewards'][s+1:e+1].sum() for s, e in zip(terminal_indices[:-1], terminal_indices[1:])])
            print('[B]', 'Dataset loaded. size: {}'.format(len(dataset['observations'])))
            print('[B]', '- number of trajectories: {}'.format(len(terminal_indices) - 1))
            print('[B]', '- average trajectory length: {:.1f} (min: {}, med: {}, max: {})'.format(
                traj_lens.mean(), traj_lens.min(), np.median(traj_lens), traj_lens.max()))
            print('[B]', '- average return: {:.3f} (min: {:.2f}, med: {:.2f}, max: {:.2f})'.format(
                traj_rets.mean(), traj_rets.min(), np.median(traj_rets), traj_rets.max()))

    def get_subtraj_dataset(self):
        # get starting indices of subtrajectories with length subtraj_len
        subtraj_indices = []
        for i in range(1, len(self.terminal_indices)):
            subtraj_indices += list(range(self.terminal_indices[i - 1] + 1, self.terminal_indices[i] - self.subtraj_len + 2))

        self.subtraj_indices = np.asarray(subtraj_indices)

        if self.verbose:
            print('[B]', 'Subtrajectories extracted. number: {}, length: {}'.format(
                len(subtraj_indices), self.subtraj_len))
