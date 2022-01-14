import torch
import torch.nn as nn


def _weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(1e-3)


def _make_layers(dims, final_relu=False):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        layers.append(nn.ReLU())
    if not final_relu:
        layers = layers[:-1]
    return nn.Sequential(*layers)


class Encoder(nn.Module):  # q_phi(z|tau)
    def __init__(self, dim_s, dim_a, dim_z, hidden_size=200, num_layers=2, num_gru_layers=4):
        super().__init__()
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_relu=True)
        gru_input_size = hidden_size + dim_a
        self.gru = nn.GRU(gru_input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=True)
        self.fc_mu = _make_layers([hidden_size, dim_z])
        self.fc_std = _make_layers([hidden_size, dim_z])

        self.apply(_weight_init)

    def forward(self, states, actions):
        # input shapes: (batch_size, subtraj_size, dim_s or dim_a)
        # output shape: tuple of (batch_size, dim_z)
        states_unrolled = states.view(-1, states.shape[-1])
        states_processed = self.fc1(states_unrolled).view(actions.shape[0], actions.shape[1], -1)
        gru_output, _ = self.gru(torch.cat((states_processed, actions), dim=-1))
        x = gru_output[:, -1, :(gru_output.shape[-1] // 2)]  # extract the first half (forward) of the last output
        return self.fc_mu(x), self.fc_std(x)


class PrimitivePolicy(nn.Module):  # pi_theta(a|s,z)
    def __init__(self, dim_s, dim_z, dim_a, hidden_size=200, num_layers=2):
        super().__init__()
        self.fc1 = _make_layers([dim_s + dim_z] + [hidden_size] * num_layers, final_relu=True)
        self.fc_mu = _make_layers([hidden_size, dim_a])
        self.fc_std = _make_layers([hidden_size, dim_a])

        self.apply(_weight_init)

    def forward(self, s, z):
        x = self.fc1(torch.cat((s, z), dim=-1))
        return self.fc_mu(x), self.fc_std(x)


class Prior(nn.Module):  # rho_omega(z|s)
    def __init__(self, dim_s, dim_z, hidden_size=200, num_layers=2):
        super().__init__()
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_relu=True)
        self.fc_mu = _make_layers([hidden_size, dim_z])
        self.fc_std = _make_layers([hidden_size, dim_z])

        self.apply(_weight_init)

    def forward(self, s):
        x = self.fc1(s)
        return self.fc_mu(x), self.fc_std(x)


class TaskPolicy(nn.Module):  # pi_psi(z|s)
    def __init__(self, dim_s, dim_z, hidden_size=256, num_layers=3):
        super().__init__()
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_relu=True)
        self.fc_mu = _make_layers([hidden_size, dim_z])
        self.fc_std = _make_layers([hidden_size, dim_z])

        self.apply(_weight_init)

    def forward(self, s):
        x = self.fc1(s)
        return self.fc_mu(x), self.fc_std(x)
