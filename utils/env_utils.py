from collections import OrderedDict
import gym
import d4rl
import metaworld
import numpy as np
import random

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


def get_metaworld_mt_env(mt_type, seed, task_agnostic):
    if mt_type == 'mt10':
        benchmark_cls = metaworld.MT10
    elif mt_type == 'mt50':
        benchmark_cls = metaworld.MT50
    else:
        raise ValueError

    benchmark = benchmark_cls(seed=seed)
    training_envs = []
    for name, env_cls in benchmark.train_classes.items():
        env = env_cls()
        task = random.choice([task for task in benchmark.train_tasks if task.env_name == name])
        env.set_task(task)
        training_envs.append(env)

    return MultiTaskEnv(training_envs, mode='vanilla' if task_agnostic else 'add-onehot')


def get_env(domain_name, task_name, seed=None, task_agnostic=False):
    env_name = get_env_name(domain_name, task_name)
    env_type = _DOMAIN_TYPES[domain_name]
    if env_type == 'd4rl':
        with SuppressStdout():
            env = gym.make(env_name)
            # env = get_antmaze_env(maze=task_name, multi_start=True, eval=True)
    elif env_type == 'metaworld':
        env = get_metaworld_mt_env(task_name, seed, task_agnostic)
    env.seed(seed)
    if hasattr(env.action_space, 'seed') and callable(env.action_space.seed):
        env.action_space.seed(seed)
    return env


class PreprocObservation(gym.ObservationWrapper):
    def __init__(self, env, fn_preproc):
        super().__init__(env)
        self.fn_preproc = fn_preproc

    def observation(self, observation):
        return self.fn_preproc(observation)


class SilentEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        with SuppressStdout():
            return self.env.step(action)

    def reset(self, **kwargs):
        with SuppressStdout():
            return self.env.reset(**kwargs)


class MultiTaskEnv(gym.Wrapper):
    """Multi-task environment wrapper.
    Code adopted from https://github.com/rlworkgroup/garage/blob/master/src/garage/envs/multi_env_wrapper.py
    """

    def __init__(self, envs, mode='vanilla'):
        assert mode in ['vanilla', 'add-onehot', 'del-onehot']
        super().__init__(envs[0])
        assert all(env.observation_space.shape == envs[0].observation_space.shape for env in envs)
        assert all(env.action_space.shape == envs[0].action_space.shape for env in envs)

        self.envs = envs
        self.num_tasks = len(envs)
        self.mode = mode
        self.active_task_index = None

        self._task_space = gym.spaces.Box(low=np.zeros(self.num_tasks), high=np.zeros(self.num_tasks))

        # set observation space
        obs_low, obs_high = self.env.observation_space.low, self.env.observation_space.high
        if self.mode == 'add-onehot':
            task_low, task_high = self._task_space.low, self._task_space.high
            obs_low, obs_high = np.concatenate([obs_low, task_low]), np.concatenate([obs_high, task_high])
        elif self.mode == 'del-onehot':
            obs_low, obs_high = obs_low[:-self.num_tasks], obs_high[:-self.num_tasks]
        self._observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def task_space(self):
        return self._task_space

    def reset(self):
        self.active_task_index = random.randint(0, self.num_tasks - 1)
        obs = self.envs[self.active_task_index].reset()
        return self._preproc_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.envs[self.active_task_index].step(action)
        if 'task_id' not in info:
            info['task_id'] = self.active_task_index
        return self._preproc_obs(obs), reward, done, info

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)
            if hasattr(env.action_space, 'seed') and callable(env.action_space.seed):
                env.action_space.seed(seed)
        return super().seed(seed)

    def _preproc_obs(self, obs):
        if self.mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        elif self.mode == 'del-onehot':
            obs = obs[:-self.num_tasks]
        return obs

    def _active_task_one_hot(self):
        """One-hot representation of active task.
        Returns:
            numpy.ndarray: one-hot representation of active task
        """
        one_hot = np.zeros(self.task_space.shape)
        index = self.active_task_index or 0
        one_hot[index] = self.task_space.high[index]
        return one_hot
