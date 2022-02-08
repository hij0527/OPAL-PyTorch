import numpy as np
import torch

from memory.buffer import Buffer
from models.opal import OPAL
from params_phase2 import parse_args
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

    # load trained OPAL
    opal = OPAL(
        dim_s=dim_s,
        dim_a=dim_a,
        dim_z=dim_z,
        device=device,
        hidden_size=args.opal_hidden_size,
        num_layers=args.opal_num_layers,
        num_gru_layers=args.opal_num_gru_layers,
    )
    opal.load_state_dict(torch.load(args.ckpt_path))

    # set up initial offline dataset
    buffer = Buffer(args.domain_name, args.task_name,
                    subtraj_len=args.subtraj_len, normalize=args.normalize, verbose=args.verbose)
    buffer.gather_data(sparse_reward=args.sparse_reward, data_dir=args.data_dir, policy_path=args.dataset_policy)
    buffer.get_subtraj_dataset()

    env = env_utils.PreprocObservation(env, buffer.normalize_observation)
    env = env_utils.SilentEnv(env)

    # training phase 2: task policy training + (optional) primitive policy finetuning 
    print('#### Training Phase 2 Start ####')
    if args.task_type == 'online':
        from models.tasks.online_model import OnlineModel
        from trainers.online_trainer import OnlineTrainer

        task_model = OnlineModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                 hidden_size=args.hidden_size, num_layers=args.num_layers)
        task_model.init_optimizer(args.lr)
        trainer = OnlineTrainer(logger=logger, phase=2, tag='online',
                                print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        trainer.train(model=task_model, env=env, device=device, longstep_len=args.subtraj_len, train_episodes=args.online_train_episodes,
                      train_start_step=args.online_train_start_step, updates_per_step=args.online_updates_per_step,
                      batch_size=args.online_batch_size, eval_freq=args.online_eval_freq, eval_episodes=args.online_eval_episodes)

    print('#### Training Phase 2 End ####')


if __name__ == "__main__":
    args = parse_args()
    main(args)
