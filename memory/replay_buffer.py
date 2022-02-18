import numpy as np
import random
import torch


class ReplayBuffer(object):
    def __init__(self, max_size, data_keys=None):
        self.max_size = max_size
        self.data_keys = data_keys
        self.buffers = []  # list of numpy.ndarray
        self.curr_idx = 0
        self.full = False

    def __len__(self):
        return self.max_size if self.full else self.curr_idx

    def keys(self):
        return self.data_keys

    def clear(self):
        self.curr_idx = 0
        self.full = False

    def add(self, transition):  # transition: tuple
        transition = [self._to_numpy(item) for item in transition]
        assert not self.data_keys or len(transition) == len(self.data_keys)

        if len(self.buffers) == 0:  # first addition
            self.buffers = [np.empty((self.max_size, *item.shape), dtype=item.dtype) for item in transition]

        for buffer, item in zip(self.buffers, transition):
            buffer[self.curr_idx] = item

        self.curr_idx = (self.curr_idx + 1) % self.max_size
        self.full = self.full or (self.curr_idx == 0)

    def add_trajectory(self, trajectory):  # trajectory: list of tuples
        if len(trajectory) > self.max_size:
            trajectory = trajectory[-self.max_size:]

        trajectory = [[self._to_numpy(item) for item in transition] for transition in trajectory]
        assert not self.data_keys or len(trajectory[0]) == len(self.data_keys)

        if len(self.buffers) == 0:  # first addition
            self.buffers = [np.empty((self.max_size, *item.shape), dtype=item.dtype) for item in trajectory[0]]

        store_indices = (np.arange(len(trajectory)) + self.curr_idx) % self.max_size

        for buffer, *item in zip(self.buffers, *trajectory):
            buffer[store_indices] = np.stack(item)

        self.curr_idx += len(trajectory)
        self.full = self.full or (self.curr_idx >= self.max_size)
        self.curr_idx %= self.max_size

    def sample(self, size=1, to_tensor=False, device=None):
        idxs = random.sample(range(len(self)), min(size, len(self)))
        samples = tuple(buffer[idxs] for buffer in self.buffers)

        if to_tensor:
            samples = tuple(self._to_tensor(item, device=device) for item in samples)

        if self.data_keys:
            samples = {k: item for k, item in zip(self.data_keys, samples)}

        return samples

    def _to_numpy(self, item):
        if isinstance(item, np.ndarray):
            return item
        elif isinstance(item, torch.Tensor):
            return item.detach().cpu().numpy()
        return np.array(item)

    def _to_tensor(self, item, dtype=torch.float32, device=None):
        if isinstance(item, torch.Tensor):
            return item.to(dtype=dtype, device=device)
        return torch.as_tensor(item, dtype=dtype, device=device)
