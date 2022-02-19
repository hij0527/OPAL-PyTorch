# Code adopted from https://github.com/nikhilbarhate99/PPO-PyTorch and https://github.com/DLR-RM/stable-baselines3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        gae_lambda=0.95,
        eps_clip=0.2,
        action_std_init=0.6,
        action_std_decay_freq=int(2.5e5),
        action_std_decay_rate=0.05,
        min_action_std=0.1,
        weight_vf=0.5,
        weight_ent=0.01,
        activation='tanh',
    ):
        super().__init__()
        self.verbose = verbose

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.eps_clip = eps_clip
        self.action_std = action_std_init
        self.action_std_decay_freq = action_std_decay_freq
        self.action_std_decay_rate = action_std_decay_rate
        self.min_action_std = min_action_std
        self.weight_vf = weight_vf
        self.weight_ent = weight_ent
        self.is_optimizer_initialized = False

        self.actor = TaskPolicy(dim_s, dim_a, hidden_size, num_layers, activation=activation)
        self.critic = ValueNetwork(dim_s, hidden_size, num_layers, activation=activation)

        self.actor_old = TaskPolicy(dim_s, dim_a, hidden_size, num_layers, activation=activation)
        self.critic_old = ValueNetwork(dim_s, hidden_size, num_layers, activation=activation)
        hard_update(self.actor_old, self.actor)
        hard_update(self.critic_old, self.critic)

        self.optimizer = None

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

    def update(self, buffer, batch_size, num_updates=80, **kwargs):
        samples = buffer.sample_all(shuffle=False, to_tensor=False)
        states_all, next_states_all = samples['observations'], samples['next_observations']
        actions_all, logprobs_all = samples['actions'], samples['logprobs']
        rewards_all, dones_all, successes_all = samples['rewards'], samples['terminals'], samples['successes']
        buffer_size = len(states_all)

        # Compute values
        values_all = np.zeros_like(rewards_all)
        next_values_all = np.zeros_like(rewards_all)
        with torch.no_grad():
            for idx in range(0, buffer_size, batch_size):
                state_batch = self._to_input(states_all[idx:idx+batch_size])
                values_all[idx:idx+batch_size] = self.critic_old(state_batch).squeeze(-1).cpu().numpy()

            # calculation of next values is only required for terminal states and the last state
            next_values_all[:-1] = values_all[1:]
            end_idxs = np.where(dones_all)[0]
            if end_idxs[-1] != buffer_size - 1:
                end_idxs = np.append(end_idxs, buffer_size - 1)  # append last idx
            for j in range(0, len(end_idxs), batch_size):
                idxs = end_idxs[j:j+batch_size]
                next_state_batch = self._to_input(next_states_all[idxs])
                next_values_all[idxs] = self.critic_old(next_state_batch).squeeze(-1).cpu().numpy()

        # Compute advantages and returns
        advantages_all = np.zeros_like(rewards_all)
        last_gae_lam = 0
        for idx in reversed(range(buffer_size)):
            # Bootstrapping is required when it is not a true done (success)
            delta = rewards_all[idx] - values_all[idx] + (1.0 - successes_all[idx]) * self.gamma * next_values_all[idx]
            last_gae_lam = delta + (1.0 - dones_all[idx]) * self.gamma * self.gae_lambda * last_gae_lam
            advantages_all[idx] = last_gae_lam

        values_all, advantages_all = values_all[:, None], advantages_all[:, None]  # match shape to critic output
        returns_all = advantages_all + values_all

        # Optimize policy for K epochs
        losses = []
        sublosses = []
        for _ in range(num_updates):
            rand_indices = np.random.permutation(buffer_size)
            for j in range(0, buffer_size, batch_size):
                idxs = rand_indices[j:j+buffer_size]
                state_batch, action_batch, logprob_batch, advantage_batch, return_batch = [
                    self._to_input(x) for x in [states_all, actions_all, logprobs_all, advantages_all, returns_all]
                ]

                # Evaluate old actions and values
                action_mean, _ = self.actor(state_batch)
                action_dist = Normal(action_mean, torch.ones_like(action_mean) * self.action_std)
                logprobs = action_dist.log_prob(action_batch).sum(-1)
                entropy = action_dist.entropy().sum(-1).mean()

                # Normalize advantages
                advantage_batch = (advantage_batch - advantage_batch.mean()) / (advantage_batch.std() + 1e-8)

                # Get ratio between old and new policies (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - logprob_batch)

                # Compute surrogate loss
                surr1 = ratios * advantage_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage_batch
                loss_surr = -torch.min(surr1, surr2).mean()

                # Compute critic loss
                state_values = self.critic(state_batch)
                loss_critic = F.mse_loss(state_values, return_batch)

                # Get final loss and step
                loss = loss_surr + self.weight_vf * loss_critic + self.weight_ent * entropy
                losses.append(loss.detach())
                sublosses.append(torch.stack([loss_surr, loss_critic, entropy]).detach())

                # TODO: early stopping by calculating reverse KL divergence

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # Copy new weights into old policy
        hard_update(self.actor_old, self.actor)
        hard_update(self.critic_old, self.critic)

        # Clear buffer
        buffer.clear()

        mean_loss = torch.stack(losses).mean()
        sublosses = torch.stack(sublosses).mean(0)

        self.update_step += 1
        return mean_loss.item(), {
            'surrogate': sublosses[0].item(),
            'critic': sublosses[1].item(),
            'entropy': sublosses[2].item(),
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

    def _to_input(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=next(self.actor.parameters()).device)
