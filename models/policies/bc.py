import torch
import torch.nn as nn

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
    ):
        super().__init__()
        self.is_optimizer_initialized = False
        self.policy = TaskPolicy(dim_s, dim_a, hidden_size, num_layers)
        self.optimizer = None
        self.update_step = 0

    def forward(self, observations):
        return self.policy(observations)

    def init_optimizers(self, lr=3e-4):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.is_optimizer_initialized = True

    def update(self, samples, **kwargs):
        state_batch, action_batch = samples['observations'], samples['actions']

        mean_a, logstd_a = self.policy(state_batch)

        loss = gaussian_nll_loss(action_batch, mean_a, logstd_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_step += 1
        return loss.item(), {
            'bc': loss.item(),
            'val_precision': (1 / logstd_a.exp().pow(2)).mean(),
        }

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
