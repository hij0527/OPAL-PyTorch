import gym
import d4rl
from d4rl.offline_env import set_dataset_path
import os

from utils.env_utils import get_env_name


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='partial', help='Task type. complete, partial, or mixed')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory to save dataset')
    args = parser.parse_args()
    args.data_dir = os.path.expanduser(args.data_dir)
    return args


def get_dataset(args):
    env = gym.make(get_env_name('kitchen', args.task))
    set_dataset_path(args.data_dir)
    return d4rl.qlearning_dataset(env)


if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args)
    print('dataset loaded. size={}'.format(len(dataset['observations'])))
