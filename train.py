import argparse
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from memory.buffer import Buffer
from models.opal import OPAL
import utils.env_utils as env_utils
import utils.python_utils as python_utils


def parse_args():
    ap = argparse.ArgumentParser('OPAL training')

    ap.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
    ap.add_argument('--gpu_id', type=str, default=None, help='GPU IDs to use. ex: 1,2 (default: None)')
    ap.add_argument('--verbose', action='store_true', help='set verbose mode')

    ap.add_argument('--results_root', type=str, default='./results', help='root directory to logs, models, videos')
    ap.add_argument('--log_dir', type=str, default='logs', help='log directory')
    ap.add_argument('--ckpt_dir', type=str, default='checkpoints', help='model (checkpoint) directory')
    ap.add_argument('--data_dir', type=str, default='./data', help='dataset directory')
    ap.add_argument('--dataset_policy', type=str, default=None, help='path to policy file for dataset collection')

    ap.add_argument('--print_freq', type=int, default=100, help='training log (stdout) frequency in steps')
    ap.add_argument('--log_freq', type=int, default=100, help='training log (tensorboard) frequency in steps')
    ap.add_argument('--save_freq', type=int, default=20, help='model save frequency in epochs')

    ap.add_argument('--domain_name', type=str, choices=env_utils.DOMAIN_NAMES, default=env_utils.DOMAIN_NAMES[0], help='environment domain name')
    ap.add_argument('--task_name', type=str, choices=env_utils.TASK_NAMES, default=env_utils.TASK_NAMES[0], help='environment task name')
    ap.add_argument('--sparse_reward', action='store_true', help='sparse reward mode')
    ap.add_argument('--dataset_size', type=int, default=int(1e6), help='size of offline dataset for phase 1')
    ap.add_argument('--subtraj_len', '-C', metavar='c', type=int, default=10, help='length of subtrajectory (c)')
    ap.add_argument('--subtraj_num', '-N', metavar='N', type=int, default=-1, help='number of subtrajectories for phase 1 (N)')
    ap.add_argument('--latent_dim', '-Z', metavar='dim_Z', type=int, default=8, help='dimension of primitive latent vector (dim(Z))')

    # model parameters
    ap.add_argument('--hidden_size', '-H', metavar='H', type=int, default=200, help='size of hidden layers (H)')
    ap.add_argument('--num_layers', type=int, default=2, help='number of hidden layers')
    ap.add_argument('--num_gru_layers', type=int, default=4, help='number of GRU layers')
    ap.add_argument('--task_hidden_size', type=int, default=256, help='size of hidden layers in task policy')
    ap.add_argument('--task_num_layers', type=int, default=2, help='number of hidden layers in task policy')

    # phase 1: primitive training
    ap.add_argument('--epochs', type=int, default=100, help='number of epochs for phase 1')
    ap.add_argument('--batch_size', type=int, default=50, help='batch size for phase 1')
    ap.add_argument('--num_workers', type=int, default=8, help='number of DataLoader workers')
    ap.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    ap.add_argument('--beta', type=float, default=0.1, help='weight of KL divergence in loss')
    ap.add_argument('--eps_kld', type=float, default=0., help='upper bound for KL divergence contraint')

    args = ap.parse_args()

    for attr in ['results_root']:
        setattr(args, attr, os.path.expanduser(getattr(args, attr)))

    return args


def main(args):
    run_id = 'opal_{}_{}'.format(args.seed, time.time())
    log_dir = os.path.join(args.results_root, args.log_dir, run_id)
    ckpt_dir = os.path.join(args.results_root, args.ckpt_dir, run_id)
    args_dir = os.path.join(args.results_root, 'args', run_id)

    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args_dir, exist_ok=True)

    def get_ckpt_name(phase, step):
        return os.path.join(ckpt_dir, 'phase{:d}_{:d}.ckpt'.format(phase, step))

    with open(os.path.join(args_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = python_utils.get_device(args.gpu_id)
    env = env_utils.get_env(args.domain_name, args.task_name)
    python_utils.seed_all(args.seed, env)

    dim_s = env.observation_space.shape[0]
    dim_a = env.action_space.shape[0]
    dim_z = args.latent_dim

    opal = OPAL(
        dim_s=dim_s,
        dim_a=dim_a,
        dim_z=dim_z,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_gru_layers=args.num_gru_layers,
        task_hidden_size=args.task_hidden_size,
        task_num_layers=args.task_num_layers,
        device=device,
    )
    opal.init_optimizers(lr=args.lr)

    writer = SummaryWriter(log_dir)

    # set up initial offline dataset
    buffer = Buffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len, verbose=args.verbose)
    buffer.gather_data(sparse_reward=args.sparse_reward, data_dir=args.data_dir, policy_path=args.dataset_policy)
    buffer.get_subtraj_dataset()
    data_loader = DataLoader(buffer, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # training phase 1: offline unsupervised primitive learning
    train_step = 0
    for epoch in range(1, args.epochs + 1):
        print('Start training phase 1 epoch {}'.format(epoch))
        epoch_loss, num_data = 0., 0
        tic = time.time()

        for i, batch in enumerate(data_loader):
            states, actions = [batch[k].to(device) for k in ['observations', 'actions']]
            loss, sublosses = opal.train_primitive(states, actions, beta=args.beta, eps_kld=args.eps_kld)
            epoch_loss += loss.item() * states.shape[0]
            num_data += states.shape[0]
            train_step += 1

            if args.print_freq > 0 and train_step % args.print_freq == 0:
                print('  step {:d} - loss: {:.6f} ({:s})'.format(
                    train_step, loss.item(),
                    ', '.join('{}: {:.6e}'.format(k, v.item()) for k, v in sublosses.items())))

            if args.log_freq > 0 and train_step % args.log_freq == 0:
                writer.add_scalar('loss_phase1', loss.item(), train_step)
                for k, v in sublosses.items():
                    writer.add_scalar('loss_phase1/{}'.format(k), v.item(), train_step)

        epoch_loss /= num_data
        print('[phase1, epoch {:d}] loss: {:.6f}, time: {:.3f}s'.format(
            epoch, epoch_loss, time.time() - tic))
        writer.add_scalar('epoch_loss_phase1', epoch_loss, epoch)

        if args.save_freq > 0 and epoch % args.save_freq == 0:
            torch.save(opal.state_dict(phase=1), get_ckpt_name(phase=1, step=epoch))

    if args.save_freq <= 0 or args.epochs % args.save_freq != 0:
        torch.save(opal.state_dict(phase=1), get_ckpt_name(phase=1, step=args.epochs))

    # temporarily added for test
    torch.save(opal.state_dict(phase=2), get_ckpt_name(phase=2, step=args.epochs))


if __name__ == "__main__":
    args = parse_args()
    main(args)
