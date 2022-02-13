# Code adopted from https://github.com/pranz24/pytorch-soft-actor-critic

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
        tau=5e-3,
        alpha=0.2,
        alpha_min=0.001,
        alpha_max=1e6,
        target_update_interval=1,
        automatic_entropy_tuning=True,
        max_entropy_range=100.,
        use_kld_penalty=False,
        action_tanh=False,
        action_scale_bias=(1, 0),
    ):
        super().__init__()
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.max_entropy_range = max_entropy_range
        self.use_kld_penalty = use_kld_penalty
        self.action_tanh = action_tanh
        self.action_scale_bias = action_scale_bias

        self.target_update_interval = target_update_interval
        self.automatic_entropy_tuning = automatic_entropy_tuning

        self.critic1 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic2 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic_target1 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        self.critic_target2 = QNetwork(dim_s, dim_a, hidden_size, num_layers=2)
        hard_update(self.critic_target1, self.critic1)
        hard_update(self.critic_target2, self.critic2)

        if self.automatic_entropy_tuning:
            self.target_entropy = -dim_a + int(self.use_kld_penalty)
            self.log_alpha = nn.Parameter(torch.zeros(1, requires_grad=True))
            self.alpha_optim = None

        self.policy = TaskPolicy(dim_s, dim_a, hidden_size, num_layers)

        self.critic_optim = None
        self.policy_optim = None

        self.update_step = 0

    def forward(self, observations):
        mean, logstd = self.policy(observations)
        if self.action_tanh:
            mean = torch.tanh(mean)
        mean = mean * self.action_scale_bias[0] + self.action_scale_bias[1]
        return mean, logstd

    def init_optimizers(self, lr_actor=3e-4, lr_critic=3e-4):
        self.critic_optim = Adam(list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr_critic)
        self.policy_optim = Adam(self.policy.parameters(), lr=lr_actor)
        if self.automatic_entropy_tuning:
            self.alpha_optim = Adam([self.log_alpha], lr=lr_actor)

    def update(self, samples, prior_model=None):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = samples
        reward_batch = reward_batch.unsqueeze(1)
        mask_batch = torch.logical_not(done_batch).to(state_batch.dtype).unsqueeze(1)

        # Update Q
        # Compute target Q values for next states
        with torch.no_grad():
            next_action, next_entropy = self._sample_action_with_entropy(next_state_batch, prior_model)
            qf1_next_target = self.critic_target1(next_state_batch, next_action)
            qf2_next_target = self.critic_target2(next_state_batch, next_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_entropy
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = self.critic1(state_batch, action_batch)
        qf2 = self.critic2(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        qf_loss = qf1_loss + qf2_loss
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()

        # Update policy
        action, entropy = self._sample_action_with_entropy(state_batch, prior_model)
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

            self.alpha = self.log_alpha.exp().clamp(min=self.alpha_min, max=self.alpha_max)
            curr_alpha = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(policy_loss.device)
            curr_alpha = torch.tensor(self.alpha)

        if (self.update_step + 1) % self.target_update_interval == 0:
            soft_update(self.critic_target1, self.critic1, self.tau)
            soft_update(self.critic_target2, self.critic2, self.tau)

        self.update_step += 1
        return policy_loss.item(), {
            'qf1': qf1_loss.item(),
            'qf2': qf2_loss.item(),
            'policy': policy_loss.item(),
            'alpha': alpha_loss.item(),
            'val_alpha': curr_alpha.item(),
            'val_entropy': entropy.mean().item(),
        }

    # Save model parameters
    def state_dict(self):
        state_dict = {
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'critic_target1': self.critic_target1.state_dict(),
            'critic_target2': self.critic_target2.state_dict(),
            'policy': self.policy.state_dict(),
            'critic_optimizer': self.critic_optim.state_dict(),
            'policy_optimizer': self.policy_optim.state_dict(),
            'curr_alpha': self.alpha,
        }
        if self.automatic_entropy_tuning:
            state_dict.update({
                'alpha_optimizer': self.alpha_optim.state_dict(),
            })
        return state_dict

    # Load model parameters
    def load_state_dict(self, state_dict):
        self.critic1.load_state_dict(state_dict['critic1'])
        self.critic2.load_state_dict(state_dict['critic2'])
        self.critic_target1.load_state_dict(state_dict['critic_target1'])
        self.critic_target2.load_state_dict(state_dict['critic_target2'])
        self.policy.load_state_dict(state_dict['policy'])
        self.critic_optim.load_state_dict(state_dict['critic_optimizer'])
        self.policy_optim.load_state_dict(state_dict['policy_optimizer'])
        self.alpha = state_dict['curr_alpha']

        if self.automatic_entropy_tuning:
            self.log_alpha.fill_(self.alpha.log() if self.alpha else 0.)
            if 'alpha_optimizer' in state_dict:
                self.alpha_optim.load_state_dict(state_dict['alpha_optimizer'])

    def _sample_action_with_entropy(self, state, prior_model=None):
        action_mean, action_logstd = self.policy(state)
        dist_action = Normal(action_mean, action_logstd.exp())
        raw_action = dist_action.rsample()
        if self.action_tanh:
            action = torch.tanh(raw_action)
        else:
            action = raw_action
        action = action * self.action_scale_bias[0] + self.action_scale_bias[1]

        entropy = dist_action.log_prob(raw_action)
        if self.action_tanh:
            entropy -= torch.log(1 - action.pow(2) + 1e-6)
        entropy = entropy.sum(-1)

        if self.use_kld_penalty:  # use KL divergence to the prior in place of entropy
            assert prior_model is not None
            with torch.no_grad():
                prior_mean, prior_logstd = prior_model(state)
            kld_penalty = gaussian_kld_loss(action_mean, action_logstd, prior_mean, prior_logstd, reduction='none')
            entropy += kld_penalty

        return action, entropy.unsqueeze(-1)  # match shape with Q output
