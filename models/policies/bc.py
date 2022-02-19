import torch
import torch.nn as nn
import torch.nn.functional as F

from models.network import TaskPolicy
from models.loss import gaussian_nll_loss


class BC(nn.Module):
    """Behavioral cloning agent"""
    def __init__(
        self,
        dim_s,
        dim_a,
        hidden_size=256,
        num_layers=3,
        loss='gauss',
    ):
        super().__init__()
        self.is_optimizer_initialized = False
        self.policy = TaskPolicy(dim_s, dim_a, hidden_size, num_layers)
        if loss == 'gauss':
            self.fn_loss = gaussian_nll_loss
        elif loss == 'mse':
            self.fn_loss = F.mse_loss
        else:
            raise ValueError('Unknown loss type: {}'.format(loss))
        self.optimizer = None
        self.update_step = 0

    def forward(self, observations):
        return self.policy(observations)

    def init_optimizers(self, lr=3e-4):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.is_optimizer_initialized = True

    def update(self, buffer, batch_size, num_updates=1, **kwargs):
        losses = []
        sublosses = []
        for _ in range(num_updates):
            samples = buffer.sample(batch_size, to_tensor=True, device=next(self.policy.parameters()).device)
            state_batch, action_batch = samples['observations'], samples['actions']

            mean_a, logstd_a = self.policy(state_batch)

            loss = self.fn_loss(action_batch, mean_a, logstd_a)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.detach())
            sublosses.append(torch.stack([(1 / logstd_a.exp().pow(2)).mean()]).detach())

        mean_loss = torch.stack(losses).mean()
        sublosses = torch.stack(sublosses).mean(0)
        subloss_dict = {
            'val_precision': sublosses[0].item(),
        }

        self.update_step += 1
        return mean_loss.item(), subloss_dict

    # Save model parameters
    def state_dict(self):
        state_dict = {
            'policy': self.policy.state_dict(),
        }
        if self.is_optimizer_initialized:
            state_dict.update({
                'optimizer': self.optimizer.state_dict(),
            })
        return state_dict

    # Load model parameters
    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
        if self.is_optimizer_initialized:
            self.optimizer.load_state_dict(state_dict['optimizer'])
