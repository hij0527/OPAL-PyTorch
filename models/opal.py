import torch
from torch.distributions import Normal
from torch.optim import Adam

from models.loss import gaussian_kld_loss, gaussian_nll_loss
from models.network import Prior, Encoder, PrimitivePolicy


class OPAL:
    def __init__(
        self,
        dim_s,
        dim_a,
        dim_z,
        device,
        hidden_size=200,
        num_layers=2,
        num_gru_layers=4,
    ):
        self.device = device

        self.encoder = Encoder(dim_s, dim_a, dim_z, hidden_size, num_layers, num_gru_layers).to(device)  # q_phi
        self.primitive_policy = PrimitivePolicy(dim_s, dim_z, dim_a, hidden_size, num_layers).to(device)  # pi_theta
        self.prior = Prior(dim_s, dim_z, hidden_size, num_layers).to(device)  # rho_omega

        self.models = [self.encoder, self.primitive_policy, self.prior]
        self.optimizer = None

    def init_optimizer(self, lr):
        self.optimizer = Adam([p for m in self.models for p in list(m.parameters())], lr=lr)

    def update(self, states, actions, beta, eps_kld=0.):  # shape of states: (batch_size, subtraj_size, dim_s)
        # shape of states: (batch_size, subtraj_size, dim_s)
        batch_size, subtraj_len = states.shape[0], states.shape[1]

        # sample z ~ q_phi(z|tau) using reparameterization
        mean_z, logstd_z = self.encoder(states, actions)
        z = Normal(mean_z, logstd_z.exp()).rsample()

        # negative log likelihood of Gaussian pi_theta (Eq. 1)
        mean_a, logstd_a = self.primitive_policy(states, z.unsqueeze(1).repeat(1, subtraj_len, 1))
        loss_nll = gaussian_nll_loss(actions, mean_a, logstd_a)

        # KL divergence between two Gaussian q_phi and rho_omega: KLD(q_phi||rho_omega) (Eq. 2)
        prior_mean_z, prior_logstd_z = self.prior(states[:, 0])
        loss_kld = gaussian_kld_loss(mean_z, logstd_z, prior_mean_z, prior_logstd_z, eps_kld)

        # update models
        loss = loss_nll + beta * loss_kld
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'nll': loss_nll.item(), 'kld': loss_kld.item()}

    def get_action_from_primitive(self, state, latent, deterministic=False, return_mean_std=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        latent = torch.as_tensor(latent, dtype=torch.float32, device=self.device).unsqueeze(0)

        mean_a, logstd_a = self.primitive_policy(state, latent)
        mean_a, logstd_a = mean_a.squeeze(0), logstd_a.squeeze(0)

        if deterministic:
            action = mean_a
        else:
            action = Normal(mean_a, logstd_a.exp()).rsample()

        if return_mean_std:
            return action, mean_a, logstd_a
        return action

    def state_dict(self):
        return {
            'encoder': self.encoder.state_dict(),
            'primitive': self.primitive_policy.state_dict(),
            'prior': self.prior.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.encoder.load_state_dict(state_dict['encoder'])
        self.primitive_policy.load_state_dict(state_dict['primitive'])
        self.prior.load_state_dict(state_dict['prior'])
        if self.optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
