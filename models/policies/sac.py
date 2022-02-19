# Code adopted from https://github.com/pranz24/pytorch-soft-actor-critic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal

from models.network import TaskPolicy, QNetwork
from models.loss import gaussian_kld_loss


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
        use_kld_penalty=False,
        action_tanh=False,
        action_scale_bias=None,
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
        self.use_kld_penalty = use_kld_penalty
        self.action_tanh = action_tanh
        self.action_scale_bias = (1, 0) if action_scale_bias is None else action_scale_bias
        self.is_optimizer_initialized = False

        self.policy = TaskPolicy(dim_s, dim_a, hidden_size, num_layers)

        self.critic1 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic2 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic_target1 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic_target2 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        hard_update(self.critic_target1, self.critic1)
        hard_update(self.critic_target2, self.critic2)

        if self.automatic_entropy_tuning:
            self.target_entropy = -dim_a + int(self.use_kld_penalty)
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
        return mean, logstd

    @property
    def alpha(self):
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp().clamp(min=self.alpha_min, max=self.alpha_max)
        return torch.tensor(self.val_alpha, dtype=torch.float32)

    def scale_action(self, action, inverse=False):
        if inverse:
            return (action - self.action_scale_bias[1]) / self.action_scale_bias[0]
        return action * self.action_scale_bias[0] + self.action_scale_bias[1]

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

            # Update Q
            # Compute target Q values for next states
            with torch.no_grad():
                next_action, next_entropy, next_kld = self._sample_action_with_entropy(next_state_batch, prior_model)
                qf1_next_target = self.critic_target1(next_state_batch, next_action)
                qf2_next_target = self.critic_target2(next_state_batch, next_action)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha.item() * next_entropy
                next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
                if self.use_kld_penalty:
                    next_q_value -= next_kld

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
            else:
                alpha_loss = torch.tensor(0.)

            losses.append(torch.tensor(0.))
            sublosses.append(torch.stack([
                policy_loss, critic_loss, alpha_loss,
                min_qf_pi.mean(), entropy.mean(),
                qf1_loss, qf2_loss,
            ]).detach())

            self.update_step += 1
            if self.update_step % self.target_update_interval == 0:
                soft_update(self.critic_target1, self.critic1, self.tau)
                soft_update(self.critic_target2, self.critic2, self.tau)

        mean_loss = torch.stack(losses).mean()
        sublosses = torch.stack(sublosses).mean(0)
        subloss_keys = [
            'policy', 'critic', 'alpha',
            'policy_min_qf_pi', 'entropy',
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
            'action_scale_bias': self.action_scale_bias,
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
        self.action_scale_bias = state_dict['action_scale_bias']
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
            action = raw_action

        entropy = dist_action.log_prob(raw_action)
        if self.action_tanh:
            entropy -= torch.log(1 - action.pow(2) + 1e-6)
        entropy = entropy.sum(-1, keepdim=True)  # match shape with Q output

        if self.use_kld_penalty:  # use KL divergence to the prior in place of entropy
            assert prior_model is not None
            with torch.no_grad():
                prior_mean, prior_logstd = prior_model(state)
            kld = gaussian_kld_loss(action_mean, action_logstd, prior_mean, prior_logstd, reduction='none')
            kld = kld.sum(-1, keepdim=True)
        else:
            kld = torch.zeros_like(entropy)

        return action, entropy, kld
