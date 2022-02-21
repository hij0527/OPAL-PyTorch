from collections import OrderedDict
import d4rl
import gym
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
    'pick-place': ['metaworld'],
})


TASK_NAMES = list(_TASK_TYPES.keys())


_ENV_NAMES = {
    ('antmaze', 'medium'): 'antmaze-medium-diverse-v2',
    ('antmaze', 'large'): 'antmaze-large-diverse-v2',
    ('kitchen', 'partial'): 'kitchen-partial-v0',
    ('kitchen', 'mixed'): 'kitchen-mixed-v0',
    ('metaworld', 'mt10'): 'metaworld-mt10-v2',
    ('metaworld', 'mt50'): 'metaworld-mt50-v2',
    ('metaworld', 'pick-place'): 'metaworld-pick-place-v2',
}


def get_env_name(domain_name, task_name):
    if domain_name not in _TASK_TYPES[task_name]:
        raise ValueError("no matching environment for domain '{}' and task '{}'".format(domain_name, task_name))
    return _ENV_NAMES[(domain_name, task_name)]


def make_antmaze_train_env(maze, sparse_reward=True, multi_start=True):
    from d4rl.locomotion import maze_env, ant

    # note: gym is using EVAL mazes
    maze_map = {
        'umaze': maze_env.U_MAZE,
        'medium': maze_env.BIG_MAZE,
        'large': maze_env.HARDEST_MAZE,
    }[maze]

    maze_kwargs = {
        'maze_map': maze_map,
        'reward_type': 'sparse' if sparse_reward else 'dense',
        'non_zero_reset': multi_start,
        'eval': True,
        'maze_size_scaling': 4.0,
        'v2_resets': True,
    }

    with SuppressStdout():
        env = ant.make_ant_maze_env(**maze_kwargs)

    return TimeoutEnv(env, 1000, name=get_env_name('antmaze', maze))


def make_metaworld_mt_env(env_name, mt_type, seed, append_task, sparse_reward):
    if mt_type == 'mt10':
        benchmark_cls = metaworld.MT10
        task_env_name = None
    elif mt_type == 'mt50':
        benchmark_cls = metaworld.MT50
        task_env_name = None
    else:
        benchmark_cls = metaworld.MT1
        task_env_name = mt_type
        if not task_env_name.endswith('-v2'):
            task_env_name += '-v2'

    if task_env_name is None:
        benchmark = benchmark_cls(seed=seed)
    else:
        benchmark = benchmark_cls(task_env_name, seed=seed)

    training_envs = []
    for name, env_cls in benchmark.train_classes.items():
        env = env_cls()
        task = random.choice([task for task in benchmark.train_tasks if task.env_name == name])
        env.set_task(task)
        training_envs.append(env)

    return MultiTaskEnv(training_envs, env_name, mode='add-onehot' if append_task else 'vanilla',
                        sparse_reward=sparse_reward)


def make_env(domain_name, task_name, seed=None, sparse_reward=True, append_task=False, eval=True, silent=True):
    env_name = get_env_name(domain_name, task_name)
    with SuppressStdout():
        if domain_name == 'antmaze':
            if eval:
                env = gym.make(env_name)
            else:
                env = make_antmaze_train_env(maze=task_name, sparse_reward=sparse_reward)
        elif domain_name == 'kitchen':
            env = gym.make(env_name)
        elif domain_name == 'metaworld':
            env = make_metaworld_mt_env(env_name, task_name, seed, append_task, sparse_reward)

    if silent:
        env = SilentEnv(env)

    env.seed(seed)
    if hasattr(env.observation_space, 'seed') and callable(env.observation_space.seed):
        env.observation_space.seed(seed)
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


class TimeoutEnv(gym.Wrapper):
    def __init__(self, env, timeout, name=None):
        super().__init__(env)
        self.curr_step = 0
        if not env.spec:
            env.spec = gym.envs.registration.EnvSpec(id=name)
        env.spec.max_episode_steps = timeout

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.curr_step += 1
        if self.curr_step == self.env.spec.max_episode_steps:
            done = True
            if 'timeout' not in info:
                info['timeout'] = True
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.curr_step = 0
        return self.env.reset(**kwargs)


class MultiTaskEnv(gym.Wrapper):
    """Multi-task environment wrapper.
    Code adopted from https://github.com/rlworkgroup/garage/blob/master/src/garage/envs/multi_env_wrapper.py
    """

    def __init__(self, envs, name, mode='vanilla', sparse_reward=False):
        assert mode in ['vanilla', 'add-onehot', 'del-onehot']
        super().__init__(envs[0])
        assert all(env.observation_space.shape == envs[0].observation_space.shape for env in envs)
        assert all(env.action_space.shape == envs[0].action_space.shape for env in envs)

        self.envs = envs
        self.num_tasks = len(envs)
        self.mode = mode
        self.sparse_reward = sparse_reward
        self.active_task_index = None
        self.is_metaworld = isinstance(self.envs[0], metaworld.envs.mujoco.mujoco_env.MujocoEnv)

        self._task_space = gym.spaces.Box(low=np.zeros(self.num_tasks), high=np.zeros(self.num_tasks))

        # set observation space
        obs_low, obs_high = self.env.observation_space.low, self.env.observation_space.high
        if self.mode == 'add-onehot':
            task_low, task_high = self._task_space.low, self._task_space.high
            obs_low, obs_high = np.concatenate([obs_low, task_low]), np.concatenate([obs_high, task_high])
        elif self.mode == 'del-onehot':
            obs_low, obs_high = obs_low[:-self.num_tasks], obs_high[:-self.num_tasks]
        self._observation_space = gym.spaces.Box(low=obs_low, high=obs_high)

        if self.is_metaworld:
            self.max_episode_steps = self.envs[0].max_path_length
        else:
            self.max_episode_steps = self.envs[0].spec.max_episode_steps

        self._spec = gym.envs.registration.EnvSpec(id=name)
        self._spec.max_episode_steps = self.max_episode_steps

        self._timestep = 0

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def base_observation_shape(self):
        return self.env.observation_space.shape

    @property
    def task_space(self):
        return self._task_space

    @property
    def spec(self):
        return self._spec

    def reset(self):
        self.active_task_index = random.randint(0, self.num_tasks - 1)
        obs = self.envs[self.active_task_index].reset()
        self._timestep = 0
        return self._preproc_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.envs[self.active_task_index].step(action)
        if 'task_id' not in info:
            info['task_id'] = self.active_task_index
        if self.is_metaworld:
            done |= bool(info['success'])
        if self.sparse_reward:
            if self.is_metaworld:
                reward = float(info['success'])
        self._timestep += 1
        if self._timestep >= self.max_episode_steps:
            done = True
        return self._preproc_obs(obs), reward, done, info

    def seed(self, seed):
        for env in self.envs:
            env.seed(seed)
            if hasattr(env.observation_space, 'seed') and callable(env.observation_space.seed):
                env.observation_space.seed(seed)
            if hasattr(env.action_space, 'seed') and callable(env.action_space.seed):
                env.action_space.seed(seed)
        return super().seed(seed)

    def get_base_observation(self, obs):
        if self.mode == 'add-onehot':
            obs = obs[:-self.num_tasks]
        return obs

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
