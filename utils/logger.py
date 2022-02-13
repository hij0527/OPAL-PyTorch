import json
import os
import time
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, args):
        self.results_root = args.results_root

        self.run_id = '{}_{}_{}_{}'.format(args.run_tag, args.domain_name, args.task_name, args.seed)
        if not args.no_timetag:
            self.run_id += '_{}'.format(time.time())
        print('Run ID: {}'.format(self.run_id))

        self.log_dir = os.path.join(self.results_root, 'logs', self.run_id)
        self.ckpt_dir = os.path.join(self.results_root, 'checkpoints', self.run_id)
        self.args_dir = os.path.join(self.results_root, 'args', self.run_id)

        os.makedirs(self.results_root, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.args_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        self.save_args(args)

    def log(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def get_ckpt_name(self, phase, step, tag=''):
        return os.path.join(self.ckpt_dir, 'phase{:d}{}_{:d}.ckpt'.format(phase, tag, step))

    def save_args(self, args):
        with open(os.path.join(self.args_dir, 'args.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
