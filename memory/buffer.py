import numpy as np
from torch.utils.data import Dataset


class Buffer(Dataset):
    def __init__(self, domain_name, task_name, subtraj_len, normalize=False, verbose=False):
        self.domain_name = domain_name
        self.task_name = task_name
        self.subtraj_len = subtraj_len
        self.normalize = normalize
        self.verbose = verbose

        self.dataset = {}
        self.terminal_indices = np.empty(0, dtype=int)
        self.subtraj_indices = np.empty(0, dtype=int)

        if self.normalize:
            self.obs_mean = np.zeros(1)
            self.obs_std = np.ones(1)

    def __getitem__(self, index):
        # return subtrajectories
        subtraj_slice = np.arange(self.subtraj_len) + self.subtraj_indices[index]
        return {
            'observations': self.dataset['observations'][subtraj_slice],
            'actions': self.dataset['actions'][subtraj_slice],
            'rewards': self.dataset['rewards'][subtraj_slice],
            'next_observations': self.dataset['next_observations'][subtraj_slice],
            'terminals': self.dataset['terminals'][subtraj_slice],
            'latents': self.dataset['latents'][index] if 'latents' in self.dataset else np.empty(0),
        }

    def __len__(self):
        return 10000
        return len(self.subtraj_indices)

    def normalize_observation(self, observation):
        if not self.normalize:
            return observation
        return (observation - self.obs_mean) / self.obs_std

    def unnormalize_observation(self, observation):
        if not self.normalize:
            return observation
        return observation * self.obs_std + self.obs_mean

    def load_data(self, sparse_reward=False, data_dir='./data', policy_path=None):
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

        # get starting indices of subtrajectories with length subtraj_len
        terminal_indices = np.where(dataset['terminals'])[0]
        terminal_indices = np.insert(terminal_indices, 0, -1)  # virtual terminal before the first transition
        if terminal_indices[-1] != len(dataset['terminals']) - 1:
            terminal_indices = np.append(terminal_indices, len(dataset['terminals']) - 1)  # treat the last transition as terminal

        subtraj_indices = []
        for i in range(1, len(terminal_indices)):
            subtraj_indices += list(range(terminal_indices[i - 1] + 1, terminal_indices[i] - self.subtraj_len + 2))

        self.dataset = dataset
        self.terminal_indices = terminal_indices
        self.subtraj_indices = np.asarray(subtraj_indices)

        if self.verbose:
            traj_lens = terminal_indices[1:] - terminal_indices[:-1]
            traj_rets = np.array([dataset['rewards'][s+1:e+1].sum() for s, e in zip(terminal_indices[:-1], terminal_indices[1:])])
            print('[B]', 'Dataset loaded. size: {}'.format(len(dataset['observations'])))
            print('[B]', '- number of trajectories: {}'.format(len(terminal_indices) - 1))
            print('[B]', '  - average trajectory length: {:.1f} (min: {}, med: {}, max: {})'.format(
                traj_lens.mean(), traj_lens.min(), np.median(traj_lens), traj_lens.max()))
            print('[B]', '  - average return: {:.3f} (min: {:.2f}, med: {:.2f}, max: {:.2f})'.format(
                traj_rets.mean(), traj_rets.min(), np.median(traj_rets), traj_rets.max()))
            print('[B]', '- number of subtrajectories (length {}): {}'.format(
                self.subtraj_len, len(subtraj_indices)))

        if self.normalize:
            obs = self.dataset['observations']
            self.obs_mean = obs.mean(0)
            self.obs_std = obs.std(0)
            self.obs_std[self.obs_std < 1e-6] = 1e-6
            self.dataset['observations'] = self.normalize_observation(obs)
            self.dataset['next_observations'] = self.normalize_observation(self.dataset['next_observations'])
            if self.verbose:
                print('[B]', 'Using normalized observations')

        if sparse_reward:
            self.dataset['rewards'] = self.dataset['rewards_sparse']
            if self.verbose:
                print('[B]', 'Using sparse reward')

    def add_latents(self, latents):
        assert(len(latents) == len(self))
        self.dataset['latents'] = latents.copy()

    def get_labeled_dataset(self):
        indices = self.subtraj_indices
        c = self.subtraj_len
        return {
            'observations': self.dataset['observations'][indices],  # s_0
            'actions': self.dataset['latents'],  # z
            'rewards': np.convolve(self.dataset['rewards'], np.ones(c), 'valid').astype(np.float32),  # sum(r_0,...,r_{c-1})
            'next_observations': self.dataset['next_observations'][indices + c - 1],  # s_c
            'terminals': self.dataset['terminals'][indices + c - 1],  # terminal_{c-1}
        }


class FixedBuffer(Dataset):
    def __init__(self, data_dict, preproc={}):
        self.data_dict = data_dict
        self.preproc = {k: (lambda x: x) for k in data_dict}
        self.preproc.update(preproc)
        self.length = len(next(iter(self.data_dict.values())))

    def __getitem__(self, idx):
        return {k: self.preproc[k](v[idx]) for k, v in self.data_dict.items()}

    def __len__(self):
        return 10000
        return self.length
