import numpy as np
import torch

from memory.replay_buffer import ReplayBuffer
from trainers.base_trainer import BaseRLTrainer, BaseHRLTrainer


class OnlineRLTrainer(BaseRLTrainer):
    def __init__(self, model, logger, device, env, eval_env=None, **kwargs):
        super().__init__(model, logger, device, env, eval_env, **kwargs)
        self.args = self._get_args(kwargs)

        self.replay_buffer = ReplayBuffer(
            max_size=self.args.max_buffer_size,
            data_keys=['observations', 'actions', 'rewards', 'next_observations', 'successes', 'logprobs'],
        )

    def _default_args(self):
        default = {
            'max_buffer_size': int(1e6),
        }
        super_default = super()._default_args()
        super_default.update(default)
        return super_default

    def train(
        self,
        train_steps=int(2.5e6),
        init_random_steps=10000,
        update_interval=1,
        updates_per_step=1,
        batch_size=256,
    ):
        tag_train_reward = '{}_eval/train_reward'.format(self.args.tag)
        self.logger.log(tag_train_reward, 0., 0)
        self.eval_and_log(0)

        episode_idx = 0
        last_step = self.env_step + train_steps
        while self.env_step < last_step:
            self.reset_timer()
            observation, episode_reward, done = self.env.reset(), 0., False
            episode_idx += 1
            episode_step = 0

            while not done and self.env_step < last_step:
                # collect data and put in replay buffer
                next_step, next_observation, reward, done, success, action, logprob = self.model.rollout_step(
                    self.env, episode_step, observation, rand=self.env_step < init_random_steps)
                self.env_step += next_step - episode_step
                episode_step = next_step
                episode_reward += reward

                self.replay_buffer.add((observation, action, reward, next_observation, success, logprob))
                self.update_model(update_interval, updates_per_step, batch_size)
                self.adjust_model_params()

            self.logger.log(tag_train_reward, episode_reward, episode_idx)
            print("[{}, episode {}] total steps: {}, episode steps: {}, reward: {:.2f}, mem: {:d}".format(
                self.args.tag, episode_idx, self.env_step, episode_step, episode_reward, len(self.replay_buffer)))

            self.eval_and_log(episode_idx)
            self.save_model(episode_idx)

        self.save_model(episode_idx, force=True)

    def update_model(self, update_interval, updates_per_step, batch_size):
        self.update_step += 1
        if updates_per_step == 0:
            return

        if self.model.on_policy:
            batch_size = update_interval
            update_kwargs = {'updates_per_step': updates_per_step}
            updates_per_step = 1
        else:
            update_kwargs = {}

        if self.update_step % update_interval == 0 and len(self.replay_buffer) >= batch_size:
            for _ in range(updates_per_step):
                samples = self.replay_buffer.sample(size=batch_size, to_tensor=True, device=self.device)
                loss, sublosses = self.model.update(samples, **update_kwargs)
            self.log_losses(self.update_step, loss, sublosses)
            if self.model.on_policy:
                self.replay_buffer.clear()

    def adjust_model_params(self):
        if hasattr(self.model, 'adjust_params'):
            self.model.adjust_params(self.update_step)


class OnlineHRLTrainer(OnlineRLTrainer, BaseHRLTrainer):
    pass
