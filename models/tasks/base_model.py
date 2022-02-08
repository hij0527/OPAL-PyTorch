import torch
from torch.distributions import Normal


class BaseModel:
    def __init__(
        self,
        opal,
        device,
        policy_type=None,
    ):
        self.opal = opal
        self.device = device
        self.policy_type = policy_type
        self.policy = None

    def train(self, dataset, **kwargs):
        raise NotImplementedError

    def get_primitive(self, state, deterministic=False, return_mean_std=False, return_logprob=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean_z, logstd_z = self.policy(state)
        mean_z, logstd_z = mean_z.squeeze(0), logstd_z.squeeze(0)

        if deterministic:
            latent = mean_z
            if return_logprob:
                print('Warning: logprob returned as 0 due to determinism')
                logprob = torch.tensor(0.)
        else:
            dist = Normal(mean_z, logstd_z.exp())
            latent = dist.rsample()
            if return_logprob:
                logprob = dist.log_prob(latent)

        ret_vals = [latent]

        if return_mean_std:
            ret_vals.append(mean_z)
            ret_vals.append(logstd_z)

        if return_logprob:
            ret_vals.append(logprob)

        return ret_vals[0] if len(ret_vals) == 1 else tuple(ret_vals)

    def get_action_from_primitive(self, state, latent, deterministic=False, return_mean_std=False):
        return self.opal.get_action_from_primitive(state, latent, deterministic, return_mean_std)

    def get_action(self, state, deterministic=True):
        latent = self.get_primitive(state, deterministic)
        return self.get_action_from_primitive(state, latent, deterministic)

    def state_dict(self):
        return {
            'policy': self.policy.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])
