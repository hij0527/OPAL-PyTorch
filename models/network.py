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


class ProbabilisticModule(nn.Module):
    """Base module with last linear layers for mean and logstd"""
    def __init__(self, dim_in, dim_out, init_method='normal'):
        super().__init__()
        self.fc_mu = _make_layers([dim_in, dim_out])
        self.fc_logstd = _make_layers([dim_in, dim_out])

        self.fc_mu.apply(partial(_weight_init, init_method=init_method))
        self.fc_logstd.apply(partial(_weight_init, init_method=init_method))

    def forward(self, x):
        mean = self.fc_mu(x)
        logstd = self.fc_logstd(x)
        return mean, logstd


class Encoder(ProbabilisticModule):  # q_phi(z|tau)
    def __init__(self, dim_s, dim_a, dim_z, hidden_size=200, num_layers=2, num_gru_layers=4):
        super().__init__(hidden_size, dim_z)
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_activation=True)
        gru_input_size = hidden_size + dim_a
        self.gru = nn.GRU(gru_input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=True)

        self.fc1.apply(_weight_init)
        self.gru.apply(partial(_weight_init, init_method='orthogonal'))

    def forward(self, states, actions):
        # input shapes: (batch_size, subtraj_size, dim_s or dim_a)
        # output shape: tuple of (batch_size, dim_z)
        states_processed = self.fc1(states)
        gru_output, _ = self.gru(torch.cat((states_processed, actions), dim=-1))
        x = gru_output[:, -1, :(gru_output.shape[-1] // 2)]  # extract the first half (forward) of the last output
        return super().forward(x)


class EncoderStateAgnostic(ProbabilisticModule):  # q_phi(z|s1, a1...ac)
    def __init__(self, dim_s, dim_a, dim_z, hidden_size=200, num_layers=2, num_gru_layers=4):
        super().__init__(hidden_size, dim_z)
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_activation=True)
        gru_input_size = hidden_size + dim_a
        self.gru = nn.GRU(gru_input_size, hidden_size, num_gru_layers, batch_first=True, bidirectional=True)

        self.fc1.apply(_weight_init)
        self.gru.apply(partial(_weight_init, init_method='orthogonal'))

    def forward(self, first_states, actions):
        # input shapes: states: (batch_size, dim_s), actions: (batch_size, subtraj_size, dim_a)
        # output shape: tuple of (batch_size, dim_z)
        states_processed = self.fc1(first_states).unsqueeze(-2).repeat_interleave(actions.shape[-2], dim=-2)
        gru_output, _ = self.gru(torch.cat((states_processed, actions), dim=-1))
        x = gru_output[:, -1, :(gru_output.shape[-1] // 2)]  # extract the first half (forward) of the last output
        return super().forward(x)


class PrimitivePolicy(ProbabilisticModule):  # pi_theta(a|s,z)
    def __init__(self, dim_s, dim_z, dim_a, hidden_size=200, num_layers=2):
        super().__init__(hidden_size, dim_a)
        self.fc1 = _make_layers([dim_s + dim_z] + [hidden_size] * num_layers, final_activation=True)

        self.fc1.apply(_weight_init)

    def forward(self, states, latents):
        x = self.fc1(torch.cat((states, latents), dim=-1))
        return super().forward(x)


class PrimitivePolicyStateAgnostic(ProbabilisticModule):  # pi_theta(a1...ac|z)
    def __init__(self, dim_z, dim_a, hidden_size=200, num_gru_layers=4):
        super().__init__(hidden_size, dim_a)
        self.gru = nn.GRU(dim_z, hidden_size, num_gru_layers, batch_first=True, bidirectional=False)

        self.gru.apply(partial(_weight_init, init_method='orthogonal'))

    def forward(self, latents):
        # input shape: (batch_size, subtraj_len, dim_z)
        # output shape: tuple of (batch_size, subtraj_len, dim_a)
        x = self.gru(latents)
        return super().forward(x)


class Prior(ProbabilisticModule):  # rho_omega(z|s)
    def __init__(self, dim_s, dim_z, hidden_size=200, num_layers=2):
        super().__init__(hidden_size, dim_z)
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers, final_activation=True)

        self.fc1.apply(_weight_init)

    def forward(self, states):
        # input shape: (..., dim_s)
        # output shape: tuple of (..., dim_z)
        x = self.fc1(states)
        return super().forward(x)


class TaskPolicy(ProbabilisticModule):  # pi_psi(z|s)
    def __init__(self, dim_s, dim_z, hidden_size=256, num_layers=3, activation='relu'):
        super().__init__(hidden_size, dim_z)
        self.fc1 = _make_layers([dim_s] + [hidden_size] * num_layers,
                                activation=activation, final_activation=True)

        self.fc1.apply(_weight_init)

    def forward(self, states):
        x = self.fc1(states)
        return super().forward(x)


class TaskPolicyMultiTask(ProbabilisticModule):  # pi_psi(z|s, i)
    def __init__(self, dim_s, dim_z, hidden_size=256, num_layers=3):
        super().__init__(hidden_size, dim_z)
        self.fc1 = _make_layers([dim_s + 1] + [hidden_size] * num_layers, final_activation=True)

        self.fc1.apply(_weight_init)

    def forward(self, s, i):
        # input shapes: (..., dim_s), (..., 1)
        # output shape: tuple of (..., dim_z)
        x = self.fc1(torch.cat((s, i), dim=-1))
        return super().forward(x)


class ValueNetwork(nn.Module):
    def __init__(self, dim_s, hidden_size=64, num_layers=2, activation='tanh'):
        super().__init__()
        self.fc = _make_layers([dim_s] + [hidden_size] * num_layers + [1], activation=activation)

        self.fc.apply(_weight_init)

    def forward(self, state):
        return self.fc(state)


class QNetwork(nn.Module):
    def __init__(self, dim_s, dim_a, hidden_size=200, num_layers=2, activation='relu'):
        super().__init__()
        self.fc = _make_layers([dim_s + dim_a] + [hidden_size] * num_layers + [1], activation=activation)

        self.fc.apply(_weight_init)

    def forward(self, states, actions):
        return self.fc(torch.cat((states, actions), dim=-1))
