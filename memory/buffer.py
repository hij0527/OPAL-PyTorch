from collections import OrderedDict
import numpy as np
from torch.utils.data import Dataset


class Buffer(Dataset):
    _ENV_TYPES = OrderedDict({
        'antmaze-medium-diverse-v0': 'd4rl',
        'antmaze-large-diverse-v0': 'd4rl',
    })
    ENVS = list(_ENV_TYPES.keys())

    def __init__(self, env, subtraj_len, verbose=False):
        self.env = env
        self.env_type = self._ENV_TYPES[env.unwrapped.spec.id]
        self.subtraj_len = subtraj_len
        self.verbose = verbose

        self.dataset = {}
        self.done_indices = np.empty(0, dtype=int)
        self.subtraj_indices = np.empty(0, dtype=int)

    def __getitem__(self, index):
        # return subtrajectories
        subtraj_slice = np.arange(self.subtraj_len) + self.subtraj_indices[index]
        return {
            'states': self.dataset['states'][subtraj_slice],
            'actions': self.dataset['actions'][subtraj_slice],
            'rewards': self.dataset['rewards'][subtraj_slice],
            'dones': self.dataset['dones'][subtraj_slice],
        }

    def __len__(self):
        return len(self.subtraj_indices)

    def gather_data(self):
        # dataset: dict of np.ndarrays with keys including 'states', 'actions', 'rewards', 'dones'
        if self.env_type == 'd4rl':
            dataset = self.env.get_dataset()
            dataset['states'] = dataset.pop('observations')
            dataset['dones'] = np.logical_or(dataset['terminals'], dataset['timeouts'])

        done_indices = np.where(dataset['dones'])[0]
        done_indices = np.insert(done_indices, 0, -1)  # virtual done before the first transition
        if done_indices[-1] != len(dataset['dones']) - 1:
            done_indices = np.append(done_indices, len(dataset['dones']) - 1)  # treat the last transition as done

        self.dataset = dataset
        self.done_indices = done_indices

        if self.verbose:
            traj_lens = done_indices[1:] - done_indices[:-1]
            traj_rets = np.array([dataset['rewards'][s+1:e+1].sum() for s, e in zip(done_indices[:-1], done_indices[1:])])
            print('[B]', 'Dataset loaded. size: {}'.format(len(dataset['states'])))
            print('[B]', '- number of trajectories: {}'.format(len(done_indices) - 1))
            print('[B]', '- average trajectory length: {:.1f} (min: {}, med: {}, max: {})'.format(
                traj_lens.mean(), traj_lens.min(), np.median(traj_lens), traj_lens.max()))
            print('[B]', '- average return: {:.3f} (min: {:.2f}, med: {:.2f}, max: {:.2f})'.format(
                traj_rets.mean(), traj_rets.min(), np.median(traj_rets), traj_rets.max()))

    def get_subtraj_dataset(self):
        # get starting indices of subtrajectories with length subtraj_len
        subtraj_indices = []
        for i in range(1, len(self.done_indices)):
            subtraj_indices += list(range(self.done_indices[i - 1] + 1, self.done_indices[i] - self.subtraj_len + 2))

        self.subtraj_indices = np.asarray(subtraj_indices)

        if self.verbose:
            print('[B]', 'Subtrajectories extracted. number: {}, length: {}'.format(
                len(subtraj_indices), self.subtraj_len))
