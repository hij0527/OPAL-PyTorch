# Code adopted from https://github.com/pranz24/pytorch-soft-actor-critic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal, kl_divergence

from models.network import TaskPolicy, QNetwork


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    target.load_state_dict(source.state_dict())


class SAC(nn.Module):
    def __init__(
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
        kld_penalty_coeff=0.,
        kld_max=100.,
        action_tanh=False,
        action_min=-1,
        action_max=1,
        adjust_action_range=False,
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.val_alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.max_entropy_range = max_entropy_range
        self.kld_penalty_coeff = kld_penalty_coeff
        self.kld_max = kld_max
        self.action_tanh = action_tanh
        self.adjust_action_range = adjust_action_range
        self.action_min = torch.as_tensor(
            data=[-torch.inf] * dim_a if action_min is None or adjust_action_range else \
                action_min if hasattr(action_min, '__iter__') else \
                [action_min] * dim_a,
            dtype=torch.float32,
        )
        self.action_max = torch.as_tensor(
            data=[torch.inf] * dim_a if action_max is None or adjust_action_range else \
                action_max if hasattr(action_max, '__iter__') else \
                [action_max] * dim_a,
            dtype=torch.float32,
        )
        self.is_optimizer_initialized = False
        self.is_alpha_updated = False

        self.policy = TaskPolicy(dim_s, dim_a, hidden_size, num_layers)

        self.critic1 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic2 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic_target1 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic_target2 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        hard_update(self.critic_target1, self.critic1)
        hard_update(self.critic_target2, self.critic2)

        if self.automatic_entropy_tuning:
            self.target_entropy = -dim_a + int(bool(self.kld_penalty_coeff))
            self.log_alpha = nn.Parameter(torch.tensor(0., requires_grad=True))
            self.alpha_optim = None

        self.policy_optim = None
        self.critic_optim = None

        self.update_step = 0

    def forward(self, observations):
        mean, logstd = self.policy(observations)
        if self.action_tanh:
            mean = torch.tanh(mean)
            mean = self.scale_action(mean)
        else:
            mean = self.clamp_action(mean)
        return mean, logstd

    @property
    def alpha(self):
        if self.automatic_entropy_tuning and self.is_alpha_updated:
            return self.log_alpha.exp().clamp(min=self.alpha_min, max=self.alpha_max)
        return torch.tensor(self.val_alpha, dtype=torch.float32)

    def scale_action(self, action, inverse=False):
        if torch.isinf(self.action_min).any() or torch.isinf(self.action_max).any():
            return action
        action_scale = ((self.action_max - self.action_min) * 0.5).to(action.device)
        action_bias = ((self.action_max + self.action_min) * 0.5).to(action.device)
        if inverse:
            return (action - action_bias) / action_scale.clamp(min=1e-6)
        return action * action_scale + action_bias

    def clamp_action(self, action):
        return action.clamp(self.action_min.to(action.device), self.action_max.to(action.device))

    def init_optimizers(self, lr_actor=3e-4, lr_critic=3e-4):
        self.policy_optim = Adam(self.policy.parameters(), lr=lr_actor)
        self.critic_optim = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr_critic)
        if self.automatic_entropy_tuning:
            self.alpha_optim = Adam([self.log_alpha], lr=lr_actor)
        self.is_optimizer_initialized = True

    def update(self, buffer, batch_size, num_updates=1, prior_model=None, **kwargs):
        losses = []
        sublosses = []
        for _ in range(num_updates):
            samples = buffer.sample(batch_size, to_tensor=True, device=next(self.policy.parameters()).device)
            state_batch, next_state_batch = samples['observations'], samples['next_observations']
            action_batch = samples['actions']
            reward_batch = samples['rewards'].unsqueeze(-1)
            mask_batch = torch.logical_not(samples['successes']).to(state_batch.dtype).unsqueeze(-1)

            if self.adjust_action_range and not prior_model:  # TODO
                self._update_action_range(action_batch)

            # Update Q
            # Compute target Q values for next states
            with torch.no_grad():
                next_action, next_entropy, next_kld = self._sample_action_with_entropy(next_state_batch, prior_model)
                qf1_next_target = self.critic_target1(next_state_batch, next_action)
                qf2_next_target = self.critic_target2(next_state_batch, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.item() * next_entropy
                next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
                next_q_value -= self.kld_penalty_coeff * next_kld

            # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1 = self.critic1(state_batch, action_batch)
            qf2 = self.critic2(state_batch, action_batch)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)

            critic_loss = qf1_loss + qf2_loss
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Update policy
            action, entropy, _ = self._sample_action_with_entropy(state_batch, prior_model)
            qf1_pi = self.critic1(state_batch, action)
            qf2_pi = self.critic2(state_batch, action)
            min_qf_pi = -torch.min(qf1_pi, qf2_pi)

            policy_loss = (self.alpha.item() * entropy + min_qf_pi).mean()
            self.policy_optim.zero_grad()
            policy_loss.backward()
            self.policy_optim.step()

            # Update alpha
            if self.automatic_entropy_tuning:
                alpha_loss = -self.log_alpha * (entropy + self.target_entropy).detach().mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.is_alpha_updated = True
            else:
                alpha_loss = torch.tensor(0., device=policy_loss.device)

            losses.append(torch.tensor(0.))
            sublosses.append(torch.stack([
                policy_loss, critic_loss, alpha_loss,
                min_qf_pi.mean(), entropy.mean(), next_kld.mean(),
                qf1_loss, qf2_loss,
            ]).detach())

            if self.update_step % self.target_update_interval == 0:
                soft_update(self.critic_target1, self.critic1, self.tau)
                soft_update(self.critic_target2, self.critic2, self.tau)

            self.update_step += 1

        mean_loss = torch.stack(losses).mean()
        sublosses = torch.stack(sublosses).mean(0)
        subloss_keys = [
            'policy', 'critic', 'alpha',
            'policy_min_qf_pi', 'entropy', 'val_kld',
            'qf1', 'qf2',
        ]
        subloss_dict = {k: v.item() for k, v in zip(subloss_keys, sublosses)}
        subloss_dict.update({
            'val_alpha': self.alpha.item(),
        })

        return mean_loss.item(), subloss_dict

    # Save model parameters
    def state_dict(self):
        state_dict = {
            'policy': self.policy.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic_target1': self.critic_target1.state_dict(),
            'critic_target2': self.critic_target2.state_dict(),
            'action_min': self.action_min,
            'action_max': self.action_max,
            'curr_alpha': self.alpha.item(),
        }
        if self.is_optimizer_initialized:
            state_dict.update({
                'policy_optimizer': self.policy_optim.state_dict(),
                'critic_optimizer': self.critic_optim.state_dict(),
            })
            if self.automatic_entropy_tuning:
                state_dict.update({
                    'alpha_optimizer': self.alpha_optim.state_dict(),
                })
        return state_dict

    # Load model parameters
    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic_target1.load_state_dict(state_dict['critic_target1'])
        self.critic_target2.load_state_dict(state_dict['critic_target2'])
        self.action_min = state_dict['action_min']
        self.action_max = state_dict['action_max']
        self.val_alpha = state_dict['curr_alpha']

        if self.is_optimizer_initialized:
            self.policy_optim.load_state_dict(state_dict['policy_optimizer'])
            self.critic_optim.load_state_dict(state_dict['critic_optimizer'])

        if self.automatic_entropy_tuning:
            self.log_alpha.data.fill_(np.log(self.val_alpha))
            if 'alpha_optimizer' in state_dict and self.is_optimizer_initialized:
                self.alpha_optim.load_state_dict(state_dict['alpha_optimizer'])

    def _sample_action_with_entropy(self, state, prior_model=None):
        action_mean, action_logstd = self.policy(state)
        dist_action = Normal(action_mean, action_logstd.exp())
        raw_action = dist_action.rsample()
        if self.action_tanh:
            action = torch.tanh(raw_action)
            action = self.scale_action(action)
        else:
            action = self.clamp_action(raw_action)

        entropy = dist_action.log_prob(raw_action)
        if self.action_tanh:
            entropy -= torch.log(1 - action.pow(2) + 1e-6)
        entropy = entropy.sum(-1, keepdim=True)  # match shape with Q output

        if self.kld_penalty_coeff:
            assert prior_model is not None
            with torch.no_grad():
                prior_mean, prior_logstd = prior_model(state)
            if self.adjust_action_range:  # TODO
                self._update_action_range(prior_mean)
            dist_prior = Normal(prior_mean, prior_logstd.exp())
            kld = kl_divergence(dist_action, dist_prior).sum(-1, keepdim=True).clamp(max=self.kld_max)
        else:
            kld = torch.zeros_like(entropy)

        return action, entropy, kld

    def _update_action_range(self, actions):
        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions)
        actions = actions.cpu()
        sample_min, sample_max = actions.min(0)[0], actions.max(0)[0]
        self.action_min = sample_min if torch.isinf(self.action_min).any() else torch.minimum(self.action_min, sample_min)
        self.action_max = sample_max if torch.isinf(self.action_max).any() else torch.maximum(self.action_max, sample_max)
