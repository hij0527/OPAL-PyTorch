import time
import torch


class BaseTrainer(object):
    def __init__(
        self,
        model,
        writer,
        get_ckpt_name,
        phase,
        tag='',
        print_freq=100,
        log_freq=100,
        save_freq=10,
    ):
        self.model = model
        self.writer = writer
        self.get_ckpt_name = get_ckpt_name
        self.phase = phase
        self.tag = tag
        self.tail_tag = '_' + tag if tag else ''
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
            print_str = '[phase{}{}, epoch {:d}] loss: {:.6f}'.format(self.phase, self.tail_tag, step, loss)
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

        loss_tag = 'phase{}{}_{}'.format(self.phase, self.tail_tag, 'epoch_loss' if is_epoch else 'loss')
        self.writer.add_scalar(loss_tag, loss, step)
        for k, v in sublosses.items():
            self.writer.add_scalar('{}/{}'.format(loss_tag, k), v, step)

    def save_model(self, epoch, force=False):
        if force or (self.save_freq > 0 and epoch % self.save_freq == 0):
            ckpt_name = self.get_ckpt_name(phase=self.phase, step=epoch, tag=self.tail_tag)
            torch.save(self.model.state_dict(phase=self.phase), ckpt_name)
