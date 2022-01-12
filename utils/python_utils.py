import numpy as np
import random
import torch


def get_device(gpu_id=None):
    if torch.cuda.is_available():
        device = 'cuda' if gpu_id is None else 'cuda:{}'.format(gpu_id)
    else:
        device = 'cpu'
    return torch.device(device)


def seed_all(seed=None, env=None):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if env is not None:
        env.seed(seed)
        if hasattr(env.action_space, 'seed') and callable(env.action_space.seed):
            env.action_space.seed(seed)
