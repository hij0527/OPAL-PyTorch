import time
import torch


class BaseTrainer(object):
    def __init__(
        self,
        logger,
        tag='',
        print_freq=100,
        log_freq=100,
        save_freq=10,
    ):
        self.logger = logger
        self.tag = tag
        self.print_freq = print_freq
        self.log_freq = log_freq
        self.save_freq = save_freq

        self.global_tic = time.time()

    def train(self):
        raise NotImplementedError

    def reset_timer(self):
        self.global_tic = time.time()

    def print_losses(self, step, loss, sublosses={}, is_epoch=False):
        if not is_epoch and (self.print_freq <= 0 or step % self.print_freq != 0):
            return

        if is_epoch:
            print_str = '[{}, epoch {:d}] loss: {:.6f}'.format(self.tag, step, loss)
        else:
            print_str = '  {} {:d} - loss: {:.6f}'.format(self.tag, step, loss)

        if sublosses:
            print_str += ' (' + ', '.join('{}: {:.6e}'.format(k, v) for k, v in sublosses.items()) + ')'

        if is_epoch:
            print_str += ', time: {:.3f}s'.format(time.time() - self.global_tic)  # TODO

        print(print_str)

    def log_losses(self, step, loss, sublosses={}, is_epoch=False):
        if not is_epoch and (self.log_freq <= 0 or step % self.log_freq != 0):
            return

        loss_tag = '{}_{}'.format(self.tag, 'epoch_loss' if is_epoch else 'loss')
        self.logger.log(loss_tag, loss, step)
        for k, v in sublosses.items():
            self.logger.log('{}/{}'.format(loss_tag, k), v, step)

    def save_model(self, model, epoch, force=False):
        if force or (self.save_freq > 0 and epoch % self.save_freq == 0):
            ckpt_name = self.logger.get_ckpt_name(step=epoch, tag=self.tag)
            torch.save(model.state_dict(), ckpt_name)
