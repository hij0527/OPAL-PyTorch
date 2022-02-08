import json
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from memory.buffer import Buffer
from models.opal import OPAL
from params import parse_args
import utils.env_utils as env_utils
import utils.python_utils as python_utils


def main(args):
    run_id = '{}_{}'.format(args.run_tag, args.seed)
    if not args.no_timetag:
        run_id += '_{}'.format(time.time())
    print('Run ID: {}'.format(run_id))

    log_dir = os.path.join(args.results_root, 'logs', run_id)
    ckpt_dir = os.path.join(args.results_root, 'checkpoints', run_id)
    args_dir = os.path.join(args.results_root, 'args', run_id)

    os.makedirs(args.results_root, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(args_dir, exist_ok=True)

    def get_ckpt_name(phase, step, tag=''):
        return os.path.join(ckpt_dir, 'phase{:d}{}_{:d}.ckpt'.format(phase, tag, step))

    with open(os.path.join(args_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    device = python_utils.get_device(args.gpu_id)
    env = env_utils.get_env(args.domain_name, args.task_name)
    python_utils.seed_all(args.seed, env)

    dim_s = np.prod(env.observation_space.shape)
    dim_a = np.prod(env.action_space.shape)
    dim_z = args.latent_dim

    if args.verbose:
        print('Dimensions: obs = {}, act = {}, latent = {}'.format(dim_s, dim_a, dim_z))

    opal = OPAL(
        dim_s=dim_s,
        dim_a=dim_a,
        dim_z=dim_z,
        device=device,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_gru_layers=args.num_gru_layers,
        task_hidden_size=args.task_hidden_size,
        task_num_layers=args.task_num_layers,
    )
    opal.init_optimizers(lr=args.lr)

    writer = SummaryWriter(log_dir)

    # set up initial offline dataset
    buffer = Buffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len, normalize=args.normalize, verbose=args.verbose)
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
            epoch_loss += loss * states.shape[0]
            num_data += states.shape[0]
            train_step += 1

            if args.print_freq > 0 and train_step % args.print_freq == 0:
                print('  step {:d} - loss: {:.6f} ({:s})'.format(
                    train_step, loss,
                    ', '.join('{}: {:.6e}'.format(k, v) for k, v in sublosses.items())))

            if args.log_freq > 0 and train_step % args.log_freq == 0:
                writer.add_scalar('loss_phase1', loss, train_step)
                for k, v in sublosses.items():
                    writer.add_scalar('loss_phase1/{}'.format(k), v, train_step)

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
