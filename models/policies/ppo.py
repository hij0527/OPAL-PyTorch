# Code adopted from https://github.com/nikhilbarhate99/PPO-PyTorch

import torch
import torch.nn as nn
from torch.distributions import Normal

from models.network import TaskPolicy, ValueNetwork


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class PPO(nn.Module):
    def __init__(
        self,
        dim_s,
        dim_a,
        verbose=False,
        hidden_size=64,
        num_layers=1,
        gamma=0.99,
        eps_clip=0.2,
        action_std_init=2.,
        action_std_decay_freq=int(2.5e5),
        action_std_decay_rate=0.05,
        min_action_std=0.1,
        weight_ent=0.002,
        policy_activation='tanh',
    ):
        super().__init__()
        self.verbose = verbose

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.action_std_decay_freq = action_std_decay_freq
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.weight_ent = weight_ent
        self.is_optimizer_initialized = False

        self.actor = TaskPolicy(dim_s, dim_a, hidden_size, num_layers, activation=policy_activation)
        self.critic = ValueNetwork(dim_s, hidden_size, num_layers, activation='tanh')

        self.actor_old = TaskPolicy(dim_s, dim_a, hidden_size, num_layers, activation=policy_activation)
        self.critic_old = ValueNetwork(dim_s, hidden_size, num_layers, activation='tanh')
        hard_update(self.actor_old, self.actor)
        hard_update(self.critic_old, self.critic)

        self.optimizer = None

        self.action_std = action_std_init
        self.lossfn_critic = nn.MSELoss()

        self.update_step = 0

    def forward(self, observations):
        """Returns inferred mean and logstd for actions"""
        action_mean, _ = self.actor_old(observations)
        action_std = max(self.action_std, 1e-6)
        return action_mean, torch.log(torch.ones_like(action_mean) * action_std)

    def init_optimizers(self, lr_actor=3e-4, lr_critic=1e-3):
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic},
        ])
        self.is_optimizer_initialized = True

    def adjust_params(self, timestep):
        # Decay action std of ouput action distribution
        if self.action_std_decay_freq > 0 and timestep % self.action_std_decay_freq == 0:
            self.action_std = max(self.action_std - self.action_std_decay_rate, self.min_action_std)
            if self.verbose:
                print('decay action_std to: {:.6f}'.format(self.action_std))

    def update(self, samples, updates_per_step=80, **kwargs):
        state_batch, action_batch, logprob_batch = samples['observations'], samples['actions'], samples['logprobs']

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_done in zip(reversed(samples['rewards']), reversed(samples['successes'])):
            if is_done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32, device=state_batch.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        rewards = rewards.unsqueeze(1)

        # Optimize policy for K epochs
        losses = []
        sublosses = []

        for _ in range(updates_per_step):
            # Evaluating old actions and values
            action_mean, _ = self.actor(state_batch)
            action_dist = Normal(action_mean, torch.ones_like(action_mean) * self.action_std)
            action_logprobs = action_dist.log_prob(action_batch).sum(-1)
            state_values = self.critic(state_batch)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(action_logprobs - logprob_batch)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            subloss_surr = -torch.min(surr1, surr2).mean()

            entropy = action_dist.entropy().sum(-1).mean()
            loss_actor = subloss_surr - self.weight_ent * entropy
            loss_critic = 0.5 * self.lossfn_critic(state_values, rewards).mean()

            # final loss of clipped objective PPO
            loss = loss_actor + loss_critic

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach())
            sublosses.append(torch.stack([loss_actor, loss_critic, subloss_surr, entropy]).detach())

        mean_loss = torch.stack(losses).mean()
        sublosses = torch.stack(sublosses).mean(0)

        # Copy new weights into old policy
        hard_update(self.actor_old, self.actor)
        hard_update(self.critic_old, self.critic)

        self.update_step += 1
        return mean_loss.item(), {
            'actor': sublosses[0].item(),
            'critic': sublosses[1].item(),
            'surrogate': sublosses[2].item(),
            'entropy': sublosses[3].item(),
            'val_action_std': self.action_std,
        }

    def state_dict(self):
        state_dict = {
            'actor': self.actor_old.state_dict(),
            'critic': self.critic_old.state_dict(),
        }
        if self.is_optimizer_initialized:
            state_dict.update({
                'optimizer': self.optimizer.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict['actor'])
        self.actor_old.load_state_dict(state_dict['actor'])
        self.critic.load_state_dict(state_dict['critic'])
        self.critic_old.load_state_dict(state_dict['critic'])
        if self.is_optimizer_initialized:
            self.optimizer.load_state_dict(state_dict['optimizer'])
