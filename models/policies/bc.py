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
        self.policy = TaskPolicy(dim_s, dim_a, hidden_size, num_layers)
        self.optimizer = None
        self.update_step = 0

    def forward(self, observations):
        return self.policy(observations)

    def init_optimizers(self, lr=3e-4):
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def update(self, samples):
        self.update_step += 1
        state_batch, action_batch = samples

        mean_a, logstd_a = self.policy(state_batch)

        loss = gaussian_nll_loss(action_batch, mean_a, logstd_a)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'bc': loss.item()}

    # Save model parameters
    def state_dict(self):
        state_dict = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        return state_dict

    # Load model parameters
    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
