import torch
from torch.distributions import Normal
from torch.optim import Adam

from models.loss import gaussian_kld_loss, gaussian_nll_loss
from models.network import Prior, Encoder, PrimitivePolicy, TaskPolicy


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
        task_hidden_size=256,
        task_num_layers=2,
    ):
        self.device = device

        self._encoder = Encoder(dim_s, dim_a, dim_z, hidden_size, num_layers, num_gru_layers).to(device)  # q_phi
        self._primitive_policy = PrimitivePolicy(dim_s, dim_z, dim_a, hidden_size, num_layers).to(device)  # pi_theta
        self._prior = Prior(dim_s, dim_z, hidden_size, num_layers).to(device)  # rho_omega
        self._task_policy = TaskPolicy(dim_s, dim_z, task_hidden_size, task_num_layers).to(device)  # pi_psi

        self._models_phase1 = [self._encoder, self._primitive_policy, self._prior]
        self._models_phase2 = [self._primitive_policy, self._task_policy]
        self._opt_phase1 = None
        self._opt_phase2 = None

    def init_optimizers(self, lr):
        self._opt_phase1 = Adam([p for m in self._models_phase1 for p in list(m.parameters())], lr=lr)
        self._opt_phase2 = Adam([p for m in self._models_phase2 for p in list(m.parameters())], lr=lr)

    def train_primitive(self, states, actions, beta, eps_kld=0.):  # shape of states: (batch_size, subtraj_size, dim_s)
        # shape of states: (batch_size, subtraj_size, dim_s)
        batch_size, subtraj_len = states.shape[0], states.shape[1]

        # sample z ~ q_phi(z|tau) using reparameterization
        mean_z, logstd_z = self._encoder(states, actions)
        z = Normal(mean_z, logstd_z.exp()).rsample()

        # negative log likelihood of Gaussian pi_theta (Eq. 1)
        mean_a, logstd_a = self._primitive_policy(states, z.unsqueeze(1).repeat(1, subtraj_len, 1))
        loss_nll = gaussian_nll_loss(actions, mean_a, logstd_a)

        # KL divergence between two Gaussian q_phi and rho_omega: KLD(q_phi||rho_omega) (Eq. 2)
        prior_mean_z, prior_logstd_z = self._prior(states[:, 0])
        loss_kld = gaussian_kld_loss(mean_z, logstd_z, prior_mean_z, prior_logstd_z, eps_kld)

        # update models
        loss = loss_nll + beta * loss_kld
        self._opt_phase1.zero_grad()
        loss.backward()
        self._opt_phase1.step()

        return loss.item(), {'nll': loss_nll.item(), 'kld': loss_kld.item()}

    def get_action(self, state, deterministic=False):
        state = torch.atleast_2d(torch.as_tensor(state, dtype=torch.float32, device=self.device))
        mean_z, logstd_z = self._task_policy(state)
        if deterministic:
            latent = mean_z
        else:
            latent = Normal(mean_z, logstd_z.exp()).sample()
        mean_a, logstd_a = self._primitive_policy(state, latent)
        if deterministic:
            action = mean_a
        else:
            action = Normal(mean_a, logstd_a.exp()).sample()
        return torch.flatten(action)

    def state_dict(self, phase):
        if phase == 1:
            return {
                'encoder': self._encoder.state_dict(),
                'primitive': self._primitive_policy.state_dict(),
                'prior': self._prior.state_dict(),
                'opt_phase1': self._opt_phase1.state_dict(),
            }
        else:
            return {
                'primitive': self._primitive_policy.state_dict(),
                'task': self._task_policy.state_dict(),
                'opt_phase2': self._opt_phase2.state_dict(),
            }

    def load_state_dict(self, state_dict, phase):
        if phase == 1:
            self._encoder.load_state_dict(state_dict['encoder'])
            self._primitive_policy.load_state_dict(state_dict['primitive'])
            self._prior.load_state_dict(state_dict['prior'])
            if self._opt_phase1:
                self._opt_phase1.load_state_dict(state_dict['opt_phase1'])
        else:
            self._primitive_policy.load_state_dict(state_dict['primitive'])
            self._task_policy.load_state_dict(state_dict['task'])
            if self._opt_phase2:
                self._opt_phase2.load_state_dict(state_dict['opt_phase2'])
