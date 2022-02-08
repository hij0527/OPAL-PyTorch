from functools import partial
import torch
import torch.nn as nn


fn_weight_init = {
    'normal': nn.init.xavier_normal_,
    'orthogonal': nn.init.orthogonal_,
}


fn_activation = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
}


def _weight_init(m, init_method='normal'):
    if isinstance(m, nn.Linear):
        fn_weight_init[init_method](m.weight.data)

        if m.bias is not None:
            nn.init.constant_(m.bias.data, 1e-3)


def _make_layers(dims, activation='relu', final_activation=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(fn_activation[activation]())
    if not final_activation:
        layers = layers[:-1]
    return nn.Sequential(*layers)


class Encoder(nn.Module):  # q_phi(z|tau)
    def __init__(self, dim_s, dim_a, dim_z, hidden_size=200, num_layers=2, num_gru_layers=4):
        super().__init__()
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_activation=True)
        gru_input_size = hidden_size + dim_a
        self.gru = nn.GRU(gru_input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=True)
        self.fc_mu = _make_layers([hidden_size, dim_z])
        self.fc_std = _make_layers([hidden_size, dim_z])

        self.gru.apply(partial(_weight_init, init_method='orthogonal'))
        self.fc_mu.apply(_weight_init)
        self.fc_std.apply(_weight_init)

    def forward(self, states, actions):
        # input shapes: (batch_size, subtraj_size, dim_s or dim_a)
        # output shape: tuple of (batch_size, dim_z)
        states_processed = self.fc1(states)
        gru_output, _ = self.gru(torch.cat((states_processed, actions), dim=-1))
        x = gru_output[:, -1, :(gru_output.shape[-1] // 2)]  # extract the first half (forward) of the last output
        mean, logstd = self.fc_mu(x), self.fc_std(x)
        return mean, logstd


class PrimitivePolicy(nn.Module):  # pi_theta(a|s,z)
    def __init__(self, dim_s, dim_z, dim_a, hidden_size=200, num_layers=2):
        super().__init__()
        self.fc1 = _make_layers([dim_s + dim_z] + [hidden_size] * num_layers, final_activation=True)
        self.fc_mu = _make_layers([hidden_size, dim_a])
        self.fc_std = _make_layers([hidden_size, dim_a])

        self.apply(_weight_init)

    def forward(self, states, latents):
        x = self.fc1(torch.cat((states, latents), dim=-1))
        mean, logstd = self.fc_mu(x), self.fc_std(x)
        return mean, logstd


class Prior(nn.Module):  # rho_omega(z|s)
    def __init__(self, dim_s, dim_z, hidden_size=200, num_layers=2):
        super().__init__()
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_activation=True)
        self.fc_mu = _make_layers([hidden_size, dim_z])
        self.fc_std = _make_layers([hidden_size, dim_z])

        self.apply(_weight_init)

    def forward(self, states):
        # input shape: (..., dim_s)
        # output shape: tuple of (..., dim_z)
        x = self.fc1(states)
        mean, logstd = self.fc_mu(x), self.fc_std(x)
        return mean, logstd


class TaskPolicy(nn.Module):  # pi_psi(z|s)
    def __init__(self, dim_s, dim_z, hidden_size=256, num_layers=3):
        super().__init__()
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_activation=True)
        self.fc_mu = _make_layers([hidden_size, dim_z])
        self.fc_std = _make_layers([hidden_size, dim_z])

        self.apply(_weight_init)

    def forward(self, states):
        x = self.fc1(states)
        mean, logstd = self.fc_mu(x), self.fc_std(x)
        return mean, logstd


class QNetwork(nn.Module):
    def __init__(self, dim_s, dim_a, hidden_size=200, num_layers=2, activation='relu'):
        super().__init__()
        self.fc = _make_layers([dim_s + dim_a] + [hidden_size] * num_layers + [1], activation=activation)

        self.apply(_weight_init)

    def forward(self, state, action):
        return self.fc(torch.cat((state, action), dim=-1))
