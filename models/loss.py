import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


def gaussian_nll_loss(x, mean, logstd):
    """Gaussian negative log likelihood loss"""
    lossfn = nn.GaussianNLLLoss(reduction='sum', full=True)
    return lossfn(x, mean, (logstd * 2).exp()) / x.shape[0]


def gaussian_kld_loss(mean1, logstd1, mean2, logstd2, eps_kld=0., reduction='mean'):
    """Bounded Kullback-Leibler divergence loss for two diagonal Gaussian distibutions"""
    kld_element = kl_divergence(Normal(mean1, logstd1.exp()), Normal(mean2, logstd2.exp()))
    kld_clamped = kld_element.sum(-1).clamp(min=eps_kld)
    if reduction == 'mean':
        return kld_clamped.mean()
    elif reduction == 'sum':
        return kld_clamped.sum()
    return kld_clamped
