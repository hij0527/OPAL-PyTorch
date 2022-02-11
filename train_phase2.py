import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from memory.buffer import SubtrajBuffer
from models.opal import OPAL
from params_phase2 import parse_args
import utils.env_utils as env_utils
from utils.logger import Logger
import utils.python_utils as python_utils


def main(args):
    args.run_tag = args.run_tag or args.task_type
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
        state_agnostic=args.opal_state_agnostic,
        unit_prior_std=args.opal_unit_prior_std,
    )
    opal.load_state_dict(torch.load(args.ckpt_path))

    # set up initial offline dataset
    buffer = SubtrajBuffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len,
                           sliding_window=args.sliding_window, normalize=args.normalize, verbose=args.verbose)
    buffer.load_data(sparse_reward=args.sparse_reward, data_dir=args.data_dir, policy_path=args.dataset_policy)

    env = env_utils.PreprocObservation(env, buffer.normalize_observation)
    env = env_utils.SilentEnv(env)

    # training phase 2: task policy training + (optional) primitive policy finetuning 
    print('#### Training Phase 2 Start ####')

    if args.task_type == 'offline':
        from memory.buffer import FixedBuffer
        from models.tasks.offline_model import OfflineModel
        from trainers.batch_trainer import BatchTrainer

        # construct a reward-labeled dataset Dr with inferred latents
        label_dataset(buffer, opal, args, device)

        # finetune primitive policy on Dr
        print('Finetuning primitive policy ...')
        opal.init_optimizer(args.lr)
        finetuner = BatchTrainer(logger=logger, phase=2, tag='finetune',
                                 print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        data_keys = ['observations', 'actions', 'latents']
        finetuner.train(model=opal, buffer=buffer, data_keys=data_keys, device=device,
                        num_epochs=args.offline_finetune_epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                        finetune=True)

        # train task policy on labeled dataset
        print('Training task policy ...')
        task_model = OfflineModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                  hidden_size=args.hidden_size, num_layers=args.num_layers)
        task_model.init_optimizer(args.lr)

        scale_reward = lambda x: np.float32((x - 0.5) * 4.0)  # TODO
        offline_buffer = FixedBuffer(buffer.get_labeled_dataset(), preproc={'rewards': scale_reward})
        trainer = BatchTrainer(logger=logger, phase=2, tag='offline',
                               print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        data_keys = ['observations', 'actions', 'rewards', 'next_observations', 'terminals']
        trainer.train(model=task_model, buffer=offline_buffer, data_keys=data_keys, device=device,
                      num_epochs=args.offline_task_epochs, batch_size=args.batch_size, num_workers=args.num_workers)

    elif args.task_type == 'imitation':
        from models.tasks.imitation_model import ImitationModel
        from trainers.batch_trainer import BatchTrainer

        # get expert demonstrations
        expert_demos = get_expert_demos(args, normalize_fn=buffer.normalize_observation)
        label_dataset(expert_demos, opal, args, device)

        # finetune primitive policy
        print('Finetuning primitive policy ...')
        opal.init_optimizer(args.lr)
        finetuner = BatchTrainer(logger=logger, phase=2, tag='finetune',
                                 print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        data_keys = ['observations', 'actions', 'latents']
        # TODO: merge expert_demos with buffer?
        finetuner.train(model=opal, buffer=expert_demos, data_keys=data_keys, device=device,
                        num_epochs=args.imitation_finetune_epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                        finetune=True)

        # train task policy via behavior cloning
        print('Training task policy ...')
        task_model = ImitationModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                    hidden_size=args.hidden_size, num_layers=args.num_layers)
        task_model.init_optimizer(args.lr)
        trainer = BatchTrainer(logger=logger, phase=2, tag='imitation',
                               print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        data_keys = ['observations', 'latents']
        batch_preproc = {'observations': (lambda x: x[:, 0, :])}  # use first states only
        trainer.train(model=task_model, buffer=expert_demos, data_keys=data_keys, device=device,
                      num_epochs=args.imitation_task_epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                      batch_preproc=batch_preproc)

    elif args.task_type == 'online':
        from models.tasks.online_model import OnlineModel
        from trainers.online_trainer import OnlineTrainer

        # train task policy online
        print('Training task policy ...')
        task_model = OnlineModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                 hidden_size=args.hidden_size, num_layers=args.num_layers)
        task_model.init_optimizer(args.lr)
        trainer = OnlineTrainer(logger=logger, phase=2, tag='online',
                                print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        trainer.train(model=task_model, env=env, device=device, longstep_len=args.subtraj_len, train_episodes=args.online_train_episodes,
                      train_start_step=args.online_train_start_step, updates_per_step=args.online_updates_per_step,
                      batch_size=args.online_batch_size, eval_freq=args.online_eval_freq, eval_episodes=args.online_eval_episodes)

    elif args.task_type == 'multitask':
        from models.tasks.multitask_model import MultitaskModel
        from trainers.multitask_trainer import MultitaskTrainer

        # train task policy via online multi-task learning
        print('Training task policy ...')
        task_model = MultitaskModel(opal, dim_s, dim_z, device, policy_type=args.policy_type,
                                    hidden_size=args.hidden_size, num_layers=args.num_layers)
        task_model.init_optimizer(args.lr)
        trainer = MultitaskTrainer(logger=logger, phase=2, tag='multitask',
                                   print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        trainer.train(model=task_model, env=env, device=device, longstep_len=args.subtraj_len, train_episodes=args.multitask_train_episodes,
                      update_freq=args.multitask_update_freq, updates_per_step=args.multitask_updates_per_step,
                      batch_size=args.batch_size, eval_freq=args.multitask_eval_freq, eval_episodes=args.multitask_eval_episodes)

    else:
        raise ValueError('Unknown task type: {}'.format(args.task_type))

    print('#### Training Phase 2 End ####')


def label_dataset(buffer: SubtrajBuffer, opal: OPAL, args, device):
    if args.load_latent_buffer:
        print('Loading labeled dataset from: {}'.format(args.load_latent_buffer))
        latent_buffer = np.load(args.load_latent_buffer)
        # TODO: sanity check: dimension, encode results
    else:
        print('Constructing labeled dataset ...')
        latent_buffer = np.empty((0, args.latent_dim), dtype=np.float32)
        sequential_loader = DataLoader(buffer, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for batch in iter(sequential_loader):
            observations, actions = [batch[k].to(device) for k in ['observations', 'actions']]
            with torch.no_grad():
                latent_batch = opal.encode(observations, actions).cpu().numpy()
            latent_buffer = np.append(latent_buffer, latent_batch, axis=0)

        if args.save_latent_buffer:
            print('Saving labeled dataset to: {}'.format(args.save_latent_buffer))
            os.makedirs(os.path.dirname(args.save_latent_buffer), exist_ok=True)
            np.save(args.save_latent_buffer, latent_buffer)

    buffer.add_latents(latent_buffer)


def get_expert_demos(args, normalize_fn):
    # TODO: 10 successful trajectories
    demos = SubtrajBuffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len,
                          sliding_window=args.sliding_window, normalize=False, verbose=args.verbose)
    # demos.load_expert_demos(sparse_reward=args.sparse_reward, data_dir=args.data_dir)
    demos.load_data(sparse_reward=args.sparse_reward, data_dir=args.data_dir, policy_path=args.dataset_policy)  # TEMPORARY
    demos.dataset['observations'] = normalize_fn(demos.dataset['observations'])
    demos.dataset['next_observations'] = normalize_fn(demos.dataset['next_observations'])
    return demos


if __name__ == "__main__":
    args = parse_args()
    main(args)
