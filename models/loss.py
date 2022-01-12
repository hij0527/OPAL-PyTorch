import torch


def gaussian_nll_loss(x, mean, logstd, batch_size):
    return (logstd + (x - mean) ** 2 / 2 / logstd.exp() ** 2).sum() / batch_size


def gaussian_kld_loss(mean1, logstd1, mean2, logstd2, batch_size, eps_kld):
    var1, var2 = logstd1.exp() ** 2, logstd2.exp() ** 2
    kld_element = (logstd2 - logstd1) - 0.5 + 0.5 * (var1 + (mean2 - mean1) ** 2) / var2
    return kld_element.view(batch_size, -1).sum(1).clamp(min=eps_kld).mean()
