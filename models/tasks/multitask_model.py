import torch

from models.tasks.base_model import BaseModel


class MultitaskModel(BaseModel):
    def __init__(
        self,
        opal,
        dim_s,
        dim_z,
        device,
        policy_type='ppo',
        hidden_size=256,
        num_layers=2,
        update_epochs=80,
    ):
        super().__init__(opal, device, policy_type)

        if policy_type == 'ppo':
            from models.policies.ppo import PPO
            self.policy = PPO(dim_s=dim_s, dim_a=dim_z, hidden_size=hidden_size,
                              num_layers=num_layers, update_epochs=update_epochs).to(device)
        else:
            raise ValueError('Unsupported policy type: {}'.format(policy_type))

    def init_optimizer(self, lr):
        self.policy.init_optimizers()   # TODO

    def update(self, samples, **kwargs):
        if self.policy_type == 'ppo':
            loss, subloss_dict = self.policy.update(samples)
        else:
            raise NotImplementedError

        return loss, subloss_dict
