import numpy as np
import os
import random
import sys
import torch


def get_device(gpu_id=None):
    if torch.cuda.is_available():
        device = 'cuda' if gpu_id is None else 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    return torch.device(device)


def seed_all(seed=None):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class SuppressStdout:
    def __enter__(self):
        self._prev_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._prev_stdout
