import numpy as np
import torch

from memory.buffer import SubtrajBuffer, RandomSamplingWrapper
from models.opal import OPAL
from params import parse_args
from trainers.batch_trainer import BatchTrainer
import utils.env_utils as env_utils
from utils.logger import Logger
import utils.python_utils as python_utils


def main(args):
    logger = Logger(args)

    device = python_utils.get_device(args.gpu_id)
    python_utils.seed_all(args.seed)
    env = env_utils.make_env(args.domain_name, args.task_name, seed=args.seed,
                             sparse_reward=True, append_task=False, eval=True)

    dim_s = np.prod(env.observation_space.shape)
    dim_a = np.prod(env.action_space.shape)
    dim_z = args.latent_dim

    if args.verbose:
        print('Dimensions: obs = {}, act = {}, latent = {}'.format(dim_s, dim_a, dim_z))

    opal = OPAL(
        dim_s=dim_s,
        dim_a=dim_a,
        dim_z=dim_z,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_gru_layers=args.num_gru_layers,
        state_agnostic=args.state_agnostic,
        unit_prior_std=args.unit_prior_std,
    ).to(device)
    opal.init_optimizers(lr=args.lr)

    # set up initial offline dataset
    buffer = SubtrajBuffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len,
                           normalize=args.normalize, verbose=args.verbose)
    buffer.load_data(data_dir=args.data_dir)
    train_buffer = RandomSamplingWrapper(buffer, size_ratio=args.num_repeat / args.subtraj_len)

    # offline unsupervised primitive learning
    print('#### OPAL Training Start ####')
    trainer = BatchTrainer(model=opal, logger=logger, device=device, env=env, tag='opal',
                           print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
    beta_schedule = (args.beta, args.beta_final if args.beta_final else args.beta)
    beta2_schedule = (args.beta2, args.beta2_final if args.beta2_final else args.beta2)
    trainer.train(train_buffer, num_epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                  train_param_schedule={'beta': beta_schedule, 'beta2': beta2_schedule},
                  truncate_normal=args.truncate_normal, eps_kld=args.eps_kld, beta=args.beta, beta2=args.beta2,
                  grad_clip_steps=args.grad_clip_steps, grad_clip_val=args.grad_clip_val)
    print('#### OPAL Training End ####')
    print('checkpoint dir: {}'.format(logger.get_ckpt_name(args.epochs, tag='opal')))


if __name__ == "__main__":
    args = parse_args()
    main(args)
