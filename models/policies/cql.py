# Code adopted from
# https://github.com/pranz24/pytorch-soft-actor-critic
# https://github.com/aviralkumar2907/CQL
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from models.policies.sac import SAC, soft_update


# TODO: match paper
# TODO: alpha never dropped below 0.001


class CQL(SAC):
    def __init__(  # TODO
        self,
        dim_s,
        dim_a,
        hidden_size=256,
        num_layers=3,
        gamma=0.99,
        tau=5e-3,
        alpha=0.2,
        target_update_interval=1,
        automatic_entropy_tuning=True,
        max_entropy_range=100.,
        use_kld_as_entropy=False,

        # CQL
        min_q_version=3,
        temp=1.0,
        min_q_weight=1.0,
        max_q_backup=False,
        deterministic_backup=True,
        num_random=10,
        with_lagrange=False,
        lagrange_thresh=0.0,
    ):
        super().__init__(
            dim_s,
            dim_a,
            hidden_size,
            num_layers,
            gamma,
            tau,
            alpha,
            target_update_interval,
            automatic_entropy_tuning,
            max_entropy_range,
            use_kld_as_entropy,
        )

        self.with_lagrange = with_lagrange
        self.deterministic_backup = deterministic_backup
        self.max_q_backup = max_q_backup
        self.min_q_ver = min_q_version
        self.temperature = temp
        self.min_q_weight = min_q_weight
        self.num_random = num_random
        self.init_bc_steps = 100  # TODO

        self.alpha_prime = torch.as_tensor(alpha, dtype=torch.float32)   # TODO

        # TODO
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = nn.Parameter(torch.zeros(1, requires_grad=True))
        # TODO

    def init_optimizers(self, lr_actor=0.0003, lr_critic=0.0003):
        super().init_optimizers(lr_actor, lr_critic)
        if self.with_lagrange:
            self.alpha_prime_optim = Adam([self.log_alpha_prime], lr=lr_critic)

    def update(self, samples):
        self.update_step += 1
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = samples
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = torch.logical_not(done_batch).to(state_batch.dtype).unsqueeze(1)

        # Update Q
        qf1 = self.critic1(state_batch, action_batch)
        qf2 = self.critic2(state_batch, action_batch)

        # Compute target Q values for next states
        with torch.no_grad():
            if self.max_q_backup:
                # select maximum Q values among N sampled actions
                next_action, _ = self._sample_multiple_actions(next_state_batch, num_actions=10)  # TODO: num_actions
                qf1_next_target = self.critic_target1(next_state_batch, next_action)
                qf2_next_target = self.critic_target2(next_state_batch, next_action)
                min_qf_next_target = torch.min(qf1_next_target.max(0)[0], qf2_next_target.max(0)[0])
            else:
                next_action, next_entropy = self._sample_action_with_entropy(next_state_batch)
                qf1_next_target = self.critic_target1(next_state_batch, next_action)
                qf2_next_target = self.critic_target2(next_state_batch, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                if not self.deterministic_backup:
                    min_qf_next_target -= self.alpha * next_entropy

            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        # CQL loss
        random_action_mul = torch.empty_like(action_batch).unsqueeze(0).repeat(self.num_random, 1, 1).uniform_(-1, 1)
        state_mul = state_batch.unsqueeze(0).repeat(self.num_random, 1, 1)
        next_state_mul = next_state_batch.unsqueeze(0).repeat(self.num_random, 1, 1)
        qf1_random = self.critic1(state_mul, random_action_mul)
        qf2_random = self.critic2(state_mul, random_action_mul)

        with torch.no_grad():
            action_mul, entropy_mul = self._sample_multiple_actions(state_batch, num_actions=self.num_random)  # TODO: equiv. to self._sample_action_with_entropy(state_dup)
            next_action_mul, next_entropy_mul = self._sample_multiple_actions(next_state_batch, num_actions=self.num_random)

        qf1_mul = self.critic1(state_mul, action_mul)
        qf2_mul = self.critic2(state_mul, action_mul)
        qf1_next_mul = self.critic1(next_state_mul, next_action_mul)
        qf2_next_mul = self.critic2(next_state_mul, next_action_mul)

        if self.min_q_ver == 3:
            random_density = -action_batch.shape[-1] * np.log(2)  # TODO
            qf1_agg = torch.cat((qf1_random - random_density, qf1_next_mul - next_entropy_mul, qf1_mul - entropy_mul), dim=1)
            qf2_agg = torch.cat((qf2_random - random_density, qf2_next_mul - next_entropy_mul, qf2_mul - entropy_mul), dim=1)
        else:
            qf1_agg = torch.cat((qf1_random, qf1.unsqueeze(1), qf1_next_mul, qf1_mul), dim=1)
            qf2_agg = torch.cat((qf2_random, qf2.unsqueeze(1), qf2_next_mul, qf2_mul), dim=1)

        min_qf1_loss = torch.logsumexp(qf1_agg / self.temperature, dim=1).mean() * self.min_q_weight * self.temperature
        min_qf2_loss = torch.logsumexp(qf2_agg / self.temperature, dim=1).mean() * self.min_q_weight * self.temperature

        min_qf1_loss -= qf1.mean() * self.min_q_weight
        min_qf2_loss -= qf2.mean() * self.min_q_weight

        if self.with_lagrange:
            min_qf1_loss = self.alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = self.alpha_prime * (min_qf2_loss - self.target_action_gap)

        qf_loss = qf1_loss + qf2_loss + min_qf1_loss + min_qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update policy
        action, entropy = self._sample_action_with_entropy(state_batch)

        if self.update_step < self.init_bc_steps:
            # behavioral cloning
            min_qf_pi = self._get_entropy(state_batch, action_batch)
        else:
            qf1_pi = self.critic1(state_batch, action)
            qf2_pi = self.critic2(state_batch, action)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = (self.alpha * entropy - min_qf_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * Entropy − Q(st,f(εt;st))]
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (entropy + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = torch.clamp(self.log_alpha.exp(), max=1e6)
            curr_alpha = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.)
            curr_alpha = torch.tensor(self.alpha)

        # Update alpha prime
        if self.with_lagrange:
            alpha_prime_loss = -0.5 * (min_qf1_loss + min_qf2_loss)
            self.alpha_prime_optim.zero_grad()
            alpha_prime_loss.backward()
            self.alpha_prime_optim.step()

            self.alpha_prime = torch.clamp(self.log_alpha_prime.exp(), max=1e6)
            curr_alpha_prime = self.alpha_prime.clone()
        else:
            alpha_prime_loss = torch.tensor(0.)
            curr_alpha_prime = torch.tensor(0.)

        if self.update_step % self.target_update_interval == 0:
            soft_update(self.critic_target1, self.critic1, self.tau)
            soft_update(self.critic_target2, self.critic2, self.tau)

        return policy_loss.item(), {
            'qf1': qf1_loss.item(),
            'qf2': qf2_loss.item(),
            'min_qf1': min_qf1_loss.item(),
            'min_qf2': min_qf2_loss.item(),
            'policy': policy_loss.item(),
            'alpha': alpha_loss.item(),
            'alpha_prime': alpha_prime_loss.item(),
            'val_alpha': curr_alpha.item(),
            'val_alpha_prime': curr_alpha_prime.item(),
        }

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict.update({
            'curr_alpha_prime': self.alpha_prime,
        })
        if self.with_lagrange:
            state_dict.update({
                'alpha_prime_optimizer': self.alpha_prime_optim.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.alpha_prime = state_dict['curr_alpha_prime']

        if self.with_lagrange:
            self.log_alpha_prime.fill_(self.alpha_prime.log() if self.alpha_prime else 0.)
            if 'alpha_prime_optimizer' in state_dict:
                self.alpha_prime_optim.load_state_dict(state_dict['alpha_prime_optimizer'])

    def _sample_multiple_actions(self, state, num_actions=10):
        """Returns num_actions batches of actions. shape: (num_actions, batch_size, dim_aciton)"""
        # TODO: tanh, action scaling
        action_mean, action_logstd = self.policy(state)
        dist_action = Normal(action_mean, action_logstd.exp())
        actions = dist_action.rsample(sample_shape=(num_actions,))
        entropy = dist_action.log_prob(actions).sum(-1)
        entropy.clamp_(-self.max_entropy_range, self.max_entropy_range)
        return actions, entropy.unsqueeze(-1)  # match shape with Q output

    def _get_entropy(self, state, action):
        # TODO: tanh, action scaling
        action_mean, action_logstd = self.policy(state)
        dist_action = Normal(action_mean, action_logstd.exp())
        entropy = dist_action.log_prob(action).sum(-1)
        entropy.clamp_(-self.max_entropy_range, self.max_entropy_range)
        return entropy.unsqueeze(-1)  # match shape with Q output
