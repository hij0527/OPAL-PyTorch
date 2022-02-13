import torch
from torch.distributions import Normal
from torch.optim import Adam

from models.loss import gaussian_kld_loss, gaussian_nll_loss
from models.network import Prior, Encoder, EncoderStateAgnostic, PrimitivePolicy, PrimitivePolicyStateAgnostic


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
        state_agnostic=False,
        unit_prior_std=False,
    ):
        self.device = device
        self.state_agnostic = state_agnostic

        # encoder (q_phi)
        if state_agnostic:
            self.encoder = EncoderStateAgnostic(dim_s, dim_a, dim_z, hidden_size, num_layers, num_gru_layers).to(device)
        else:
            self.encoder = Encoder(dim_s, dim_a, dim_z, hidden_size, num_layers, num_gru_layers).to(device)

        # primitive policy (pi_theta)
        if state_agnostic:
            self.primitive_policy = PrimitivePolicyStateAgnostic(dim_z, dim_a, hidden_size, num_gru_layers).to(device)
        else:
            self.primitive_policy = PrimitivePolicy(dim_s, dim_z, dim_a, hidden_size, num_layers).to(device)

        # prior (rho_omega)
        self.prior = Prior(dim_s, dim_z, hidden_size, num_layers, unit_prior_std=unit_prior_std).to(device)

        self.models = [self.encoder, self.primitive_policy, self.prior]
        self.optimizer = None

        self.update_step = 0

    def init_optimizer(self, lr):
        self.optimizer = Adam([p for m in self.models for p in list(m.parameters())], lr=lr)

    def update(self, samples, **kwargs):
        finetune = kwargs.pop('finetune', False)

        if finetune:
            return self._finetune(samples, **kwargs)
        else:
            return self._update(samples, **kwargs)

    def encode(self, states, actions):
        mean_z, _ = self.encoder(states[:, 0] if self.state_agnostic else states, actions)
        return mean_z

    def get_action_from_primitive(self, state, latent, deterministic=False, return_mean_std=False):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        latent = torch.as_tensor(latent, dtype=torch.float32, device=self.device).unsqueeze(0)

        if self.state_agnostic:
            mean_a, logstd_a = self.primitive_policy(latent)
        else:
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

    def _update(self, samples, **kwargs):  # shape of states: (batch_size, subtraj_size, dim_s)
        states, actions = samples
        # shape of states: (batch_size, subtraj_size, dim_s)
        batch_size, subtraj_len = states.shape[0], states.shape[1]

        # sample z ~ q_phi(z|tau) using reparameterization
        mean_z, logstd_z = self.encoder(states[:, 0] if self.state_agnostic else states, actions)
        eps = torch.randn_like(mean_z)
        max_eps = kwargs.get('truncate_normal', None)
        if max_eps:
            eps = torch.fmod(eps, max_eps)  # truncate values outside of (-max_eps, max_eps)
        z = mean_z + logstd_z.exp() * eps

        # negative log likelihood of Gaussian pi_theta (Eq. 1)
        # mean, logstd shape: (batch_size, c, dim_a)
        if self.state_agnostic:
            mean_a, logstd_a = self.primitive_policy(z.unsqueeze(1).repeat(1, subtraj_len, 1))
        else:
            mean_a, logstd_a = self.primitive_policy(states, z.unsqueeze(1).repeat(1, subtraj_len, 1))
        loss_nll = gaussian_nll_loss(actions, mean_a, logstd_a)

        # KL divergence between two Gaussian q_phi and rho_omega: KLD(q_phi||rho_omega) (Eq. 2)
        prior_mean_z, prior_logstd_z = self.prior(states[:, 0])
        eps_kld = kwargs.get('eps_kld', 0.)
        loss_kld = gaussian_kld_loss(mean_z, logstd_z, prior_mean_z, prior_logstd_z, eps_kld)

        # additional regularization
        beta2 = kwargs.get('beta2', 0.)
        if beta2:
            unit_mean, unit_logstd = torch.zeros_like(mean_z), torch.zeros_like(logstd_z)
            loss_reg = gaussian_kld_loss(mean_z, logstd_z, unit_mean, unit_logstd)
        else:
            loss_reg = torch.zeros_like(loss_kld)

        # update models
        beta = kwargs.get('beta', 0.1)
        loss = loss_nll + beta * loss_kld + beta2 * loss_reg
        self.optimizer.zero_grad()
        loss.backward()

        grad_clip_steps = kwargs.get('grad_clip_steps', 0)
        if grad_clip_steps < 0 or self.update_step < grad_clip_steps:
            # clip gradients
            clip_val = kwargs.get('grad_clip_val', 10.)
            for model in self.models:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
        self.optimizer.step()

        self.update_step += 1
        return loss.item(), {
            'nll': loss_nll.item(),
            'kld': loss_kld.item(),
            'reg': loss_reg.item(),
            'beta': beta,
            'beta2': beta2,
            'precision': (1 / logstd_z.exp() ** 2).mean(),
        }

    def _finetune(self, samples, **kwargs):
        states, actions, latents = samples
        batch_size, subtraj_len = states.shape[0], states.shape[1]

        # behavior cloning; same as minimizing negative log likelihood (Eq. 3)
        mean_a, logstd_a = self.primitive_policy(states, latents.unsqueeze(1).repeat(1, subtraj_len, 1))
        loss = gaussian_nll_loss(actions, mean_a, logstd_a)

        # update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), {'bc': loss.item()}
