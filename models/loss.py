import numpy as np
import torch
import torch.nn as nn


def gaussian_nll_loss(x, mean, logstd):
    """Gaussian negative log likelihood loss"""
    lossfn = nn.GaussianNLLLoss(reduction='sum', full=True)
    return lossfn(x, mean, (logstd * 2).exp()) / x.shape[0]
