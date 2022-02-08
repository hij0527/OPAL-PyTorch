import numpy as np
import torch

from memory.buffer import Buffer
from models.opal import OPAL
from params import parse_args
from trainers.batch_trainer import BatchTrainer
import utils.env_utils as env_utils
from utils.logger import Logger
import utils.python_utils as python_utils


def main(args):
    logger = Logger(args)

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
    )
    opal.init_optimizer(lr=args.lr)

    # set up initial offline dataset
    buffer = Buffer(args.domain_name, args.task_name,
                    subtraj_len=args.subtraj_len, normalize=args.normalize, verbose=args.verbose)
    buffer.gather_data(sparse_reward=args.sparse_reward, data_dir=args.data_dir, policy_path=args.dataset_policy)
    buffer.get_subtraj_dataset()

    # training phase 1: offline unsupervised primitive learning
    print('#### Training Phase 1 Start ####')
    trainer = BatchTrainer(logger=logger, phase=1, tag='',
                           print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
    data_keys = ['observations', 'actions']
    trainer.train(model=opal, buffer=buffer, data_keys=data_keys, device=device,
                  num_epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                  beta=args.beta, eps_kld=args.eps_kld)
    print('#### Training Phase 1 End ####')


if __name__ == "__main__":
    args = parse_args()
    main(args)
