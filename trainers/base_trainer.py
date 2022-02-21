from argparse import Namespace
import numpy as np
import time
import torch


class BaseTrainer:
    def __init__(self, model, logger, device, **kwargs):
        self.args = self._get_args(kwargs)
        self.model = model
        self.logger = logger

        self.global_tic = time.time()
        self.device = device
        self.update_step = 0

    def _get_args(self, kwargs):
        args = self._default_args()
        args.update(kwargs)
        return Namespace(**args)

    def _default_args(self):
        return {
            'tag': '',
            'print_freq': 100,
            'log_freq': 100,
            'save_freq': 10,
            'eval_freq': 1,
            'eval_num': 1,
        }

    def train_model(self, **kwargs):
        raise NotImplementedError

    def eval_model(self, **kwargs):
        raise NotImplementedError

    def reset_timer(self):
        self.global_tic = time.time()

    def print_losses(self, step, loss, sublosses={}, force=False, is_epoch=False):
        if not force and (self.args.print_freq <= 0 or step % self.args.print_freq != 0):
            return

        if is_epoch:
            print_str = '[{}, epoch {:d}] loss: {:.6f}'.format(self.args.tag, step, loss)
        else:
            print_str = '  {} {:d} - loss: {:.6f}'.format(self.args.tag, step, loss)

        if sublosses:
            print_str += ' (' + ', '.join('{}: {:.6e}'.format(k, v) for k, v in sublosses.items()) + ')'

        if is_epoch:
            print_str += ', time: {:.3f}s'.format(time.time() - self.global_tic)  # TODO

        print(print_str)

    def log_losses(self, step, loss, sublosses={}, force=False, is_epoch=False):
        if not force and (self.args.log_freq <= 0 or step % self.args.log_freq != 0):
            return

        loss_tag = '{}_{}'.format(self.args.tag, 'epoch_loss' if is_epoch else 'loss')
        self.logger.log(loss_tag, loss, step)
        for k, v in sublosses.items():
            self.logger.log('{}/{}'.format(loss_tag, k), v, step)

    def save_model(self, epoch, force=False):
        if not force and (self.args.save_freq <= 0 or epoch % self.args.save_freq != 0):
            return

        ckpt_name = self.logger.get_ckpt_name(step=epoch, tag=self.args.tag)
        torch.save(self.model.state_dict(), ckpt_name)

    @torch.no_grad()
    def eval_and_log(self, epoch, force=False, **kwargs):
        if not force and (self.args.eval_freq <= 0 or epoch % self.args.eval_freq != 0):
            return

        eval_results = self.eval_model(**kwargs)
        eval_tag = '{}_eval'.format(self.args.tag)
        for k, v in eval_results.items():
            self.logger.log('{}/{}'.format(eval_tag, k), v, epoch)

        print('[{}, eval] '.format(self.args.tag) + \
            ', '.join('{}: {:.6e}'.format(k, v) for k, v in eval_results.items()))


class BaseRLTrainer(BaseTrainer):
    def __init__(self, model, logger, device, env, eval_env=None, **kwargs):
        super().__init__(model, logger, device, **kwargs)
        self.args = self._get_args(kwargs)

        self.env = env
        self.eval_env = env if eval_env is None else eval_env
        self.env_step = 0

    def eval_model(self, **kwargs):
        episode_rewards = []
        episode_successes = []
        episode_lengths = []
        for _  in range(self.args.eval_num):
            observation, ep_reward, done = self.eval_env.reset(), 0., False
            step = 0
            while not done:
                next_step, next_observation, reward, done, success, *extra = self.model.rollout_step(
                    self.eval_env, step, observation, rand=False, deterministic=True)
                ep_reward += reward
                step = next_step
                observation = next_observation
            episode_rewards.append(ep_reward)
            episode_successes.append(float(success))
            episode_lengths.append(step)
        episode_rewards = np.asarray(episode_rewards)
        episode_successes = np.asarray(episode_successes)
        episode_lengths = np.asarray(episode_lengths)
        return {
            'reward': episode_rewards.mean(),
            'rew_min': episode_rewards.min(),
            'rew_max': episode_rewards.max(),
            'success': episode_successes.mean(),
            'ep_len': episode_lengths.mean(),
        }


class BaseHRLTrainer(BaseRLTrainer):
    def _default_args(self):
        default = {
            'longstep_len': 10,
            'multitask': False,
        }
        super_default = super()._default_args()
        super_default.update(default)
        return super_default
