# Code adopted from
# https://github.com/aviralkumar2907/CQL
# https://github.com/pranz24/pytorch-soft-actor-critic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, Uniform

from models.policies.sac import SAC, soft_update


# TODO: match paper


class CQL(SAC):
    def __init__(  # TODO
        self,
        dim_s,
        dim_a,
        hidden_size=256,
        num_layers=3,
        gamma=0.99,
        tau=1e-2,
        alpha=0.2,
        alpha_min=0.,
        alpha_max=1e6,
        target_update_interval=1,
        automatic_entropy_tuning=True,
        max_entropy_range=100.,
        use_kld_penalty=False,
        action_tanh=False,
        action_scale_bias=None,

        # CQL
        reg_type='H',  # 'H', 'rho', or 'var'
        temp=1.0,
        max_q_backup=False,
        deterministic_backup=True,
        num_random=10,
        with_lagrange=False,
        lagrange_thresh=0.0,
        alpha_prime=5.0,
        alpha_prime_min=0.,
        alpha_prime_max=1e6,
        init_bc_steps=40000,
        adjust_action_range=False,
    ):
        super().__init__(
            dim_s=dim_s,
            dim_a=dim_a,
            hidden_size=hidden_size,
            num_layers=num_layers,
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
            target_update_interval=target_update_interval,
            automatic_entropy_tuning=automatic_entropy_tuning,
            max_entropy_range=max_entropy_range,
            use_kld_penalty=False,
            action_tanh=action_tanh,
            action_scale_bias=action_scale_bias,
        )

        self.val_alpha_prime = alpha_prime
        self.alpha_prime_min = alpha_prime_min
        self.alpha_prime_max = alpha_prime_max
        self.with_lagrange = with_lagrange
        self.deterministic_backup = deterministic_backup
        self.max_q_backup = max_q_backup
        self.reg_type = reg_type
        self.temperature = temp
        self.num_random = num_random
        self.init_bc_steps = init_bc_steps
        self.adjust_action_range = adjust_action_range

        if adjust_action_range:
            self.action_range = None
        else:
            self.action_range = self.scale_action(torch.stack([-torch.ones(dim_a), torch.ones(dim_a)]))

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = nn.Parameter(torch.tensor(0., requires_grad=True))

    @property
    def alpha_prime(self):
        if self.with_lagrange:
            return self.log_alpha_prime.exp().clamp(min=self.alpha_prime_min, max=self.alpha_prime_max)
        return torch.tensor(self.val_alpha_prime, dtype=torch.float32)

    def init_optimizers(self, lr_actor=3e-5, lr_critic=3e-4):
        super().init_optimizers(lr_actor, lr_critic)
        if self.with_lagrange:
            self.alpha_prime_optim = Adam([self.log_alpha_prime], lr=lr_critic)

    def update(self, samples, **kwargs):
        state_batch, action_batch, next_state_batch = samples['observations'], samples['actions'], samples['next_observations']
        reward_batch = samples['rewards'].unsqueeze(-1)
        mask_batch = torch.logical_not(samples['successes']).to(state_batch.dtype).unsqueeze(-1)

        if self.adjust_action_range:
            self._update_action_range(action_batch)

        # Update Q
        qf1 = self.critic1(state_batch, action_batch)
        qf2 = self.critic2(state_batch, action_batch)

        # Compute target Q values for next states
        with torch.no_grad():
            if self.max_q_backup:
                # select maximum Q values among N sampled actions
                num_actions = 10
                next_action, _ = self._sample_multiple_actions(next_state_batch, num_actions=num_actions)
                next_state_mul = next_state_batch.unsqueeze(0).repeat(num_actions, 1, 1)
                qf1_next_target = self.critic_target1(next_state_mul, next_action).max(0)[0]
                qf2_next_target = self.critic_target2(next_state_mul, next_action).max(0)[0]
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
            else:
                next_action, next_entropy, _ = self._sample_action_with_entropy(next_state_batch)
                qf1_next_target = self.critic_target1(next_state_batch, next_action)
                qf2_next_target = self.critic_target2(next_state_batch, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                if not self.deterministic_backup:
                    min_qf_next_target -= self.alpha.item() * next_entropy

            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)

        # CQL loss
        with torch.no_grad():
            state_mul = state_batch.unsqueeze(0).repeat(self.num_random, 1, 1)
            # random_action_mul = torch.empty_like(action_batch).unsqueeze(0).repeat(self.num_random, 1, 1).uniform_(-1, 1)
            random_action_mul = self._sample_random_actions((self.num_random,) + action_batch.shape[:-1], action_batch.device)
            new_action1_mul, new_entropy1_mul = self._sample_multiple_actions(state_batch, num_actions=self.num_random)
            new_action2_mul, new_entropy2_mul = self._sample_multiple_actions(next_state_batch, num_actions=self.num_random)

        qf1_rand_mul = self.critic1(state_mul, random_action_mul)
        qf2_rand_mul = self.critic2(state_mul, random_action_mul)
        qf1_new1_mul = self.critic1(state_mul, new_action1_mul)
        qf2_new1_mul = self.critic2(state_mul, new_action1_mul)
        qf1_new2_mul = self.critic1(state_mul, new_action2_mul)  # note: it is correct to use state, not next_state
        qf2_new2_mul = self.critic2(state_mul, new_action2_mul)  # https://github.com/aviralkumar2907/CQL/issues/4

        if self.reg_type == 'H':
            random_density = -action_batch.shape[-1] * np.log(2)
            qf1_agg = torch.cat((qf1_rand_mul - random_density, qf1_new2_mul - new_entropy2_mul, qf1_new1_mul - new_entropy1_mul))
            qf2_agg = torch.cat((qf2_rand_mul - random_density, qf2_new2_mul - new_entropy2_mul, qf2_new1_mul - new_entropy1_mul))
        else:
            qf1_agg = torch.cat((qf1_rand_mul, qf1.unsqueeze(0), qf1_new2_mul, qf1_new1_mul))
            qf2_agg = torch.cat((qf2_rand_mul, qf2.unsqueeze(0), qf2_new2_mul, qf2_new1_mul))

        min_qf1_loss = torch.logsumexp(qf1_agg / self.temperature, dim=0).mean() * self.temperature - qf1.mean()
        min_qf2_loss = torch.logsumexp(qf2_agg / self.temperature, dim=0).mean() * self.temperature - qf2.mean()

        if self.with_lagrange:
            min_qf1_loss -= self.target_action_gap
            min_qf2_loss -= self.target_action_gap

        qf_loss = qf1_loss + qf2_loss + self.alpha_prime.item() * (min_qf1_loss + min_qf2_loss)
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update policy
        action, entropy, _ = self._sample_action_with_entropy(state_batch)

        if self.update_step < self.init_bc_steps:
            # behavioral cloning
            min_qf_pi = self._get_entropy(state_batch, action_batch)
        else:
            qf1_pi = self.critic1(state_batch, action)
            qf2_pi = self.critic2(state_batch, action)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha.item() * entropy - min_qf_pi).mean()
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -self.log_alpha * (entropy + self.target_entropy).detach().mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        else:
            alpha_loss = torch.tensor(0.)

        # Update alpha prime
        if self.with_lagrange:
            alpha_prime_loss = -0.5 * self.alpha_prime * (min_qf1_loss + min_qf2_loss).detach()
            self.alpha_prime_optim.zero_grad()
            alpha_prime_loss.backward()
            self.alpha_prime_optim.step()
        else:
            alpha_prime_loss = torch.tensor(0.)

        if (self.update_step + 1) % self.target_update_interval == 0:
            soft_update(self.critic_target1, self.critic1, self.tau)
            soft_update(self.critic_target2, self.critic2, self.tau)

        self.update_step += 1
        return policy_loss.item(), {
            'qf1': qf1_loss.item(),
            'qf2': qf2_loss.item(),
            'min_qf1': min_qf1_loss.item(),
            'min_qf2': min_qf2_loss.item(),
            'policy': policy_loss.item(),
            'alpha': alpha_loss.item(),
            'alpha_prime': alpha_prime_loss.item(),
            'val_alpha': self.alpha.item(),
            'val_alpha_prime': self.alpha_prime.item(),
            'val_entropy': entropy.mean().item(),
        }

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({
            'curr_alpha_prime': self.alpha_prime.item(),
        })
        if self.with_lagrange and self.is_optimizer_initialized:
            state_dict.update({
                'alpha_prime_optimizer': self.alpha_prime_optim.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.val_alpha_prime = state_dict['curr_alpha_prime']

        if self.with_lagrange:
            self.log_alpha_prime.data.fill_(np.log(self.val_alpha_prime))
            if 'alpha_prime_optimizer' in state_dict and self.is_optimizer_initialized:
                self.alpha_prime_optim.load_state_dict(state_dict['alpha_prime_optimizer'])

    def _sample_multiple_actions(self, state, num_actions=10):
        """Returns num_actions batches of actions. shape: (num_actions, batch_size, dim_aciton)"""
        action_mean, action_logstd = self.policy(state)
        dist_action = Normal(action_mean, action_logstd.exp())
        raw_actions = dist_action.rsample(sample_shape=(num_actions,))
        if self.action_tanh:
            actions = torch.tanh(raw_actions)
            actions = self.scale_action(actions)
        else:
            actions = raw_actions

        entropy = dist_action.log_prob(raw_actions)
        if self.action_tanh:
            entropy -= torch.log(1 - actions.pow(2) + 1e-6)
        entropy = entropy.sum(-1, keepdim=True)

        return actions, entropy

    def _get_entropy(self, state, action):
        action = self.scale_action(action, inverse=True)
        if self.action_tanh:
            assert action.abs().max() < 1 + 1e-6
            raw_action = torch.atanh(action * (1 - 1e-6))
        else:
            raw_action = action
        action_mean, action_logstd = self.policy(state)
        dist_action = Normal(action_mean, action_logstd.exp())

        entropy = dist_action.log_prob(raw_action)
        if self.action_tanh:
            entropy -= torch.log(1 - action.pow(2) + 1e-6)
        entropy = entropy.sum(-1, keepdim=True)

        return entropy

    def _update_action_range(self, actions):
        sample_min, sample_max = actions.min(0)[0], actions.max(0)[0]
        if self.action_range is None:
            self.action_range = torch.stack([sample_min, sample_max])
        else:
            self.action_range = torch.stack([
                torch.minimum(self.action_range[0], sample_min),
                torch.maximum(self.action_range[1], sample_max)
            ])

    def _sample_random_actions(self, batch_shape, device):
        assert self.action_range is not None
        return Uniform(*self.action_range).sample(batch_shape).to(device)
