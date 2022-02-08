import torch

from models.tasks.base_model import BaseModel


class OnlineModel(BaseModel):
    def __init__(
        self,
        opal,
        dim_s,
        dim_z,
        device,
        policy_type='sac',
        hidden_size=256,
        num_layers=2,
    ):
        super().__init__(opal, device, policy_type)
        if policy_type == 'sac':
            from models.policies.sac import SAC
            self.policy = SAC(dim_s=dim_s, dim_a=dim_z, hidden_size=hidden_size,
                              num_layers=num_layers, use_kld_as_entropy=True).to(device)
        else:
            raise ValueError('Unsupported online policy type: {}'.format(policy_type))

    def init_optimizer(self, lr):
        self.policy.init_optimizers()   #TODO

    def update(self, samples, **kwargs):
        if self.policy_type == 'sac':
            loss, subloss_dict = self.policy.update(samples, prior_model=self.opal.prior)
        else:
            raise NotImplementedError

        return loss, subloss_dict
