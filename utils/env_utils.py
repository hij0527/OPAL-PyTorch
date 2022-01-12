from collections import OrderedDict
import gym
import d4rl

from utils.python_utils import SuppressStdout


_DOMAIN_TYPES = OrderedDict({
    'antmaze': 'd4rl',
    'kitchen': 'd4rl',
    'metaworld': 'metaworld',
})


DOMAIN_NAMES = list(_DOMAIN_TYPES.keys())


_TASK_TYPES = OrderedDict({
    'medium': ['antmaze'],
    'large': ['antmaze'],
    'partial': ['kitchen'],
    'mixed': ['kitchen'],
    'mt10': ['metaworld'],
    'mt50': ['metaworld'],
})


TASK_NAMES = list(_TASK_TYPES.keys())


_ENV_NAMES = {
    ('antmaze', 'medium'): 'antmaze-medium-diverse-v2',
    ('antmaze', 'large'): 'antmaze-large-diverse-v2',
    ('kitchen', 'partial'): 'kitchen-partial-v0',
    ('kitchen', 'mixed'): 'kitchen-mixed-v0',
    ('metaworld', 'mt10'): '',
    ('metaworld', 'mt50'): '',
}


def get_env_name(domain_name, task_name):
    if domain_name not in _TASK_TYPES[task_name]:
        raise ValueError("no matching environment for domain '{}' and task '{}'".format(domain_name, task_name))
    return _ENV_NAMES[(domain_name, task_name)]


def get_antmaze_env(maze, multi_start=True, eval=False):
    from d4rl.locomotion import maze_env, ant
    from d4rl.locomotion.wrappers import NormalizedBoxEnv

    maze_map = {
        'umaze': maze_env.U_MAZE,
        'medium': maze_env.BIG_MAZE,
        'large': maze_env.HARDEST_MAZE,
        'umaze_eval': maze_env.U_MAZE_EVAL,
        'medium_eval': maze_env.BIG_MAZE_EVAL,
        'large_eval': maze_env.HARDEST_MAZE_EVAL
    }[maze]

    return NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze_map, maze_size_scaling=4.0, non_zero_reset=multi_start, eval=eval))


def get_env(domain_name, task_name):
    env_name = get_env_name(domain_name, task_name)
    env_type = _DOMAIN_TYPES[domain_name]
    if env_type == 'd4rl':
        with SuppressStdout():
            return gym.make(env_name)
            # return get_antmaze_env(maze=task_name, multi_start=True, eval=True)
    elif env_type == 'metaworld':
        raise NotImplementedError
