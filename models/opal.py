import torch
from torch.optim import Adam

from models.network import Prior, Encoder, PrimitivePolicy, TaskPolicy


class OPAL:
    def __init__(
        self,
        dim_s,
        dim_a,
        dim_z,
        hidden_size,
        num_layers,
        num_gru_layers,
        task_hidden_size,
        task_num_layers,
        device,
    ):
        self._encoder = Encoder(dim_s, dim_a, dim_z, hidden_size, num_layers, num_gru_layers).to(device)  # q_phi
        self._primitive_policy = PrimitivePolicy(dim_s, dim_z, dim_a, hidden_size, num_layers).to(device)  # pi_theta
        self._prior = Prior(dim_s, dim_z, hidden_size, num_layers).to(device)  # rho_omega
        self._task_policy = TaskPolicy(dim_s, dim_z, task_hidden_size, task_num_layers).to(device)  # pi_psi

        self._opt_enc = None
        self._opt_primitive = None
        self._opt_prior = None
        self._opt_task = None

    def init_optimizers(self, lr):
        self._opt_enc = Adam(self._encoder.parameters(), lr=lr)
        self._opt_primitive = Adam(self._primitive_policy.parameters(), lr=lr)
        self._opt_prior = Adam(self._prior.parameters(), lr=lr)
        self._opt_task = Adam(self._task_policy.parameters(), lr=lr)

    def train_primitive(self, states, actions, beta):  # shape of states: (batch_size, subtraj_size, dim_s)
        # sample z ~ q_phi(z|tau) using reparameterization
        mean_z, logstd_z = self._encoder(states, actions)
        eps = torch.randn_like(mean_z)
        z = mean_z + logstd_z.exp() * eps

        # negative log likelihood of Gaussian pi_theta (Eq. 1)
        mean_a, logstd_a = self._primitive_policy(states.view(-1, states.shape[-1]), z.repeat(states.shape[1], 1))
        actions_unrolled = actions.view(-1, actions.shape[-1])
        loss_nll = (logstd_a + (actions_unrolled - mean_a) ** 2 / 2 / logstd_a.exp() ** 2).sum()
        loss_nll /= states.shape[0]

        # KL divergence between two Gaussian q_phi and rho_omega: KLD(q_phi||rho_omega) (Eq. 2)
        prior_mean_z, prior_logstd_z = self._prior(states[:, 0])
        var_z, prior_var_z = logstd_z.exp() ** 2, prior_logstd_z.exp() ** 2
        loss_kld = ((prior_logstd_z - logstd_z) - 0.5 \
                    + 0.5 * (var_z + (prior_mean_z - mean_z) ** 2) / prior_var_z).sum()
        loss_kld /= states.shape[0]

        # update models
        loss = loss_nll + beta * loss_kld
        self._opt_enc.zero_grad()
        self._opt_primitive.zero_grad()
        self._opt_prior.zero_grad()
        loss.backward()
        self._opt_enc.step()
        self._opt_primitive.step()
        self._opt_prior.step()

        return loss, {'nll': loss_nll, 'kld': loss_kld}

    def state_dict(self, is_phase1=False):
        state_dict = {
            'encoder': self._encoder.state_dict(),
            'primitive': self._primitive_policy.state_dict(),
            'prior': self._prior.state_dict(),
            'opt_enc': self._opt_enc.state_dict(),
            'opt_primitive': self._opt_primitive.state_dict(),
            'opt_prior': self._opt_prior.state_dict(),
        }

        if not is_phase1:
            state_dict.update({
                'task': self._task_policy.state_dict(),
                'opt_task': self._opt_task.state_dict(),
            })

        return state_dict

    def load_state_dict(self, state_dict, is_phase1=False):
        self._encoder.load_state_dict(state_dict['encoder'])
        self._primitive_policy.load_state_dict(state_dict['primitive'])
        self._prior.load_state_dict(state_dict['prior'])
        self._opt_enc.load_state_dict(state_dict['opt_enc'])
        self._opt_primitive.load_state_dict(state_dict['opt_primitive'])
        self._opt_prior.load_state_dict(state_dict['opt_prior'])

        if not is_phase1:
            self._task_policy.load_state_dict(state_dict['task'])
            self._opt_task.load_state_dict(state_dict['opt_task'])
