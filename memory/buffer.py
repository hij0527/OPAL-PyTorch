import argparse
import numpy as np
from torch.utils.data import Dataset


class Buffer(Dataset):
    def __init__(self, domain_name, task_name, normalize=False, sparse_reward=False, verbose=False):
        self.domain_name = domain_name
        self.task_name = task_name
        self.normalize = normalize
        self.sparse_reward = sparse_reward
        self.verbose = verbose

        self.dataset = {}

        if self.normalize:
            self.obs_mean = None
            self.obs_std = None

    def __getitem__(self, index):
        return {
            'observations': self.dataset['observations'][index],
            'actions': self.dataset['actions'][index],
            'rewards': self.dataset['rewards'][index],
            'next_observations': self.dataset['next_observations'][index],
            'terminals': self.dataset['terminals'][index],
            'successes': self.dataset['successes'][index],
        }

    def __len__(self):
        return len(self.dataset['observations'])

    def normalize_observation(self, observation):
        if not self.normalize or self.obs_mean is None:
            return observation
        return (observation - self.obs_mean) / self.obs_std

    def unnormalize_observation(self, observation):
        if not self.normalize or self.obs_mean is None:
            return observation
        return observation * self.obs_std + self.obs_mean

    def load_data(self, data_dir='./data', policy_path=None):
        if self.domain_name == 'antmaze':
            from utils.antmaze_dataset import get_dataset
            dataset = get_dataset(argparse.Namespace(
                task=self.task_name,
                policy=policy_path,
                data_dir=data_dir,
                multi_start=True,
                multigoal=True,
                noisy=True,
            ), verbose=self.verbose)
        elif self.domain_name == 'kitchen':
            from utils.kitchen_dataset import get_dataset
            dataset = get_dataset(argparse.Namespace(
                task=self.task_name,
                policy=policy_path,
                data_dir=data_dir,
            ), verbose=self.verbose)
        elif self.domain_name == 'metaworld':
            from utils.metaworld_dataset import get_dataset
            dataset = get_dataset(argparse.Namespace(
                task='pick-place',
                policy=policy_path,
                data_dir=data_dir,
                noisy=True,
            ), verbose=self.verbose)
        else:
            raise NotImplementedError

        self.dataset = self._preprocess(dataset)

        if self.verbose:
            print('[B]', 'Dataset loaded. size: {}'.format(len(self.dataset['observations'])))

    def load_expert_demos(self, data_dir='./data', policy_path=None, num_demos=10):
        if self.domain_name == 'antmaze':
            from utils.antmaze_dataset import get_dataset
            dataset = get_dataset(argparse.Namespace(
                task=self.task_name,
                policy=policy_path,
                data_dir=data_dir,
                multi_start=True,
                multigoal=True,
                noisy=False,
                expert_demos=num_demos,
            ), verbose=self.verbose)
        else:
            raise NotImplementedError

        self.dataset = self._preprocess(dataset)

        if self.verbose:
            print('[B]', 'Expert demos: {}'.format(len(self.dataset['observations'])))

    def _preprocess(self, dataset):
        if 'successes' not in dataset:
            if 'timeouts' in dataset:
                dataset['successes'] = np.logical_and(
                    dataset['terminals'],
                    np.logical_not(dataset['timeouts'])
                )
            elif 'rewards_sparse' in dataset:
                dataset['successes'] = np.logical_and(
                    dataset['terminals'],
                    dataset['rewards_sparse'] > 0
                )
            else:
                dataset['successes'] = dataset['terminals']

        if self.normalize:
            if self.obs_mean is None:
                obs = dataset['observations']
                self.obs_mean = obs.mean(0)
                self.obs_std = obs.std(0)
                self.obs_std[self.obs_std < 1e-6] = 1e-6
            dataset['observations'] = self.normalize_observation(dataset['observations'])
            dataset['next_observations'] = self.normalize_observation(dataset['next_observations'])
            if self.verbose:
                print('[B]', 'Using normalized observations')

        if self.sparse_reward and 'rewards_sparse' in dataset:
            dataset['rewards'] = dataset['rewards_sparse']
            if self.verbose:
                print('[B]', 'Using sparse reward')

        return dataset


class SubtrajBuffer(Buffer):
    def __init__(self,
        domain_name,
        task_name,
        subtraj_len,
        sliding_window_step=0,
        normalize=False,
        sparse_reward=False,
        verbose=False,
    ):
        super().__init__(domain_name, task_name, normalize, sparse_reward, verbose)
        self.subtraj_len = subtraj_len
        self.sliding_window_step = sliding_window_step or subtraj_len

        self.terminal_indices = np.empty(0, dtype=int)
        self.subtraj_indices = np.empty(0, dtype=int)

    def __getitem__(self, index):
        """Get subtrajectories"""
        subtraj_slice = np.arange(self.subtraj_len) + self.subtraj_indices[index]
        item = super().__getitem__(subtraj_slice)
        if 'latents' in self.dataset:
            item['latents'] = self.dataset['latents'][index]
        return item

    def __len__(self):
        return len(self.subtraj_indices)

    def add_latents(self, latents):
        assert len(latents) == len(self.subtraj_indices)
        self.dataset['latents'] = latents.copy()

    def get_labeled_dataset(self):
        assert 'latents' in self.dataset
        indices = self.subtraj_indices
        c = self.subtraj_len
        return {
            'observations': self.dataset['observations'][indices],  # s_0
            'latents': self.dataset['latents'],  # z
            'rewards': np.convolve(self.dataset['rewards'], np.ones(c), 'valid')[indices].astype(np.float32),  # sum(r_0,...,r_{c-1})
            'next_observations': self.dataset['next_observations'][indices + c - 1],  # s_c
            'terminals': self.dataset['terminals'][indices + c - 1],  # terminal_{c-1}
            'successes': self.dataset['successes'][indices + c - 1],  # success_{c-1}
        }

    def _preprocess(self, dataset):
        dataset = super()._preprocess(dataset)
        return self._split_trajectories(dataset)

    def _split_trajectories(self, dataset):
        # get subtrajectories
        terminal_indices = np.where(dataset['terminals'])[0]
        terminal_indices = np.insert(terminal_indices, 0, -1)  # virtual terminal before the first transition
        if terminal_indices[-1] != len(dataset['terminals']) - 1:
            # treat the last transition as terminal
            terminal_indices = np.append(terminal_indices, len(dataset['terminals']) - 1)

        # get starting indices of subtrajectories with length subtraj_len
        subtraj_indices = []
        for i in range(1, len(terminal_indices)):
            # start_indices will be empty for too short trajectories (length < subtraj_len)
            start_indices = range(
                terminal_indices[i - 1] + 1,
                terminal_indices[i] - self.subtraj_len + 2,
                self.sliding_window_step
            )
            subtraj_indices += list(start_indices)
        subtraj_indices = np.asarray(subtraj_indices)

        self.terminal_indices = terminal_indices
        self.subtraj_indices = subtraj_indices

        if self.verbose:
            traj_lens = terminal_indices[1:] - terminal_indices[:-1]
            traj_rets = np.array(
                [dataset['rewards'][s+1:e+1].sum() for s, e in zip(terminal_indices[:-1], terminal_indices[1:])]
            )
            print('[B]', '- number of trajectories: {}'.format(len(terminal_indices) - 1))
            print('[B]', '  - average trajectory length: {:.1f} (min: {}, med: {}, max: {})'.format(
                traj_lens.mean(), traj_lens.min(), np.median(traj_lens), traj_lens.max()))
            print('[B]', '  - average return: {:.3f} (min: {:.2f}, med: {:.2f}, max: {:.2f})'.format(
                traj_rets.mean(), traj_rets.min(), np.median(traj_rets), traj_rets.max()))
            print('[B]', '- number of subtrajectories (length {}): {}'.format(
                self.subtraj_len, len(subtraj_indices)))

        return dataset


class FixedBuffer(Dataset):
    def __init__(self, data_dict, preproc={}):
        self.data_dict = data_dict
        self.preproc = {k: (lambda x: x) for k in data_dict}
        self.preproc.update(preproc)
        self.length = len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx):
        return {k: self.preproc[k](v[idx]) for k, v in self.data_dict.items()}

    def __len__(self):
        return self.length


class RandomSamplingWrapper:
    """Wrapper of buffer for random sampling"""
    def __init__(self, buffer, size_ratio=1.):
        self.buffer = buffer
        self.size_ratio = size_ratio

    def __len__(self):
        return int(len(self.buffer) * self.size_ratio)

    def __getitem__(self, index):
        rand_index = np.random.randint(len(self.buffer))
        return self.buffer[rand_index]
