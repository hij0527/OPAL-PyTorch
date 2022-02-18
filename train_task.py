import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from memory.buffer import SubtrajBuffer
from models.agent import HRLAgent
from models.opal import OPAL
from params_task import parse_args
import utils.env_utils as env_utils
from utils.logger import Logger
import utils.python_utils as python_utils


def main(args):
    args.run_tag = args.run_tag or args.task_type
    logger = Logger(args)

    device = python_utils.get_device(args.gpu_id)
    python_utils.seed_all(args.seed)
    multitask = args.task_type == 'multitask'
    env = env_utils.make_env(args.domain_name, args.task_name, seed=args.seed, eval=False,
                             sparse_reward=args.sparse_reward, append_task=multitask)
    eval_env = env_utils.make_env(args.domain_name, args.task_name, seed=args.seed + 100, eval=True,
                                  sparse_reward=True, append_task=multitask)

    dim_s = np.prod(env.observation_space.shape)
    dim_s_opal = dim_s - np.prod(env.task_space.shape) if multitask else dim_s
    dim_a = np.prod(env.action_space.shape)
    dim_z = args.latent_dim

    if args.verbose:
        print('Dimensions: obs = {} (opal: {}), act = {}, latent = {}'.format(dim_s, dim_s_opal, dim_a, dim_z))

    # load trained OPAL
    opal = OPAL(
        dim_s=dim_s_opal,
        dim_a=dim_a,
        dim_z=dim_z,
        hidden_size=args.opal_hidden_size,
        num_layers=args.opal_num_layers,
        num_gru_layers=args.opal_num_gru_layers,
        state_agnostic=args.opal_state_agnostic,
        unit_prior_std=args.opal_unit_prior_std,
    ).to(device)
    opal.load_state_dict(torch.load(args.opal_ckpt))

    # set up initial offline dataset
    buffer = SubtrajBuffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len,
                           normalize=args.normalize, sparse_reward=args.sparse_reward, verbose=args.verbose)
    buffer.load_data(data_dir=args.data_dir)

    env = env_utils.PreprocObservation(env, buffer.normalize_observation)
    env = env_utils.SilentEnv(env)

    # task policy training + (optional) primitive policy finetuning
    print('#### Task Policy Training Start ####')

    if args.task_type == 'offline':
        from memory.buffer import FixedBuffer, RandomSamplingWrapper
        from trainers.batch_trainer import BatchTrainer, BatchHRLTrainer

        # construct a reward-labeled dataset Dr with inferred latents
        label_dataset(buffer, opal, args, device)
        finetune_buffer = RandomSamplingWrapper(buffer, size_ratio=args.num_repeat / args.subtraj_len)

        # finetune primitive policy on Dr
        print('Finetuning primitive policy ...')
        opal.init_optimizers(lr=args.lr)
        finetuner = BatchTrainer(opal, logger, device, tag='opal_finetune',
                                 print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        finetuner.train(finetune_buffer, num_epochs=args.offline_finetune_epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        finetune=True)

        # train task policy on labeled dataset
        # scale_reward = lambda x: np.float32((x - 0.5) * 4.0)  # TODO: from CQL
        offline_buffer = FixedBuffer(buffer.get_labeled_dataset())  #, preproc={'rewards': scale_reward})

        print('Training task policy ...')
        model = HRLAgent(args.policy_type, opal, longstep_len=args.subtraj_len,
                         dim_s=dim_s, dim_z=dim_z,
                         hidden_size=args.hidden_size, num_layers=args.num_layers,
                         policy_param_str=args.offline_policy_param_str).to(device)
        model.init_optimizers()  # TODO
        trainer = BatchHRLTrainer(model, logger, device, env, eval_env=eval_env, tag='offline',
                                  print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq,
                                  eval_freq=args.eval_freq, eval_num=args.eval_num,
                                  longstep_len=args.subtraj_len)
        trainer.train(offline_buffer, num_epochs=args.offline_task_epochs,
                      batch_size=args.batch_size, num_workers=args.num_workers)

    elif args.task_type == 'imitation':
        from memory.buffer import RandomSamplingWrapper
        from trainers.batch_trainer import BatchTrainer, BatchHRLTrainer

        # get expert demonstrations
        demos = SubtrajBuffer(args.domain_name, args.task_name, subtraj_len=args.subtraj_len,
                              sliding_window_step=1, normalize=False,
                              sparse_reward=args.sparse_reward, verbose=args.verbose)
        demos.load_expert_demos(data_dir=args.data_dir, policy_path=args.imitation_expert_policy,
                                num_demos=args.imitation_num_demos)
        demos.dataset['observations'] = buffer.normalize_observation(demos.dataset['observations'])
        demos.dataset['next_observations'] = buffer.normalize_observation(demos.dataset['next_observations'])
        label_dataset(demos, opal, args, device)
        demos = RandomSamplingWrapper(demos, size_ratio=1.)

        # finetune primitive policy
        print('Finetuning primitive policy ...')
        opal.init_optimizers(lr=args.lr)
        finetuner = BatchTrainer(opal, logger, device, tag='opal_finetune',
                                 print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq)
        finetuner.train(demos, num_epochs=args.imitation_finetune_epochs,
                        batch_size=args.batch_size, num_workers=args.num_workers,
                        finetune=True)

        # train task policy via behavioral cloning
        print('Training task policy ...')
        model = HRLAgent(args.policy_type, opal, longstep_len=args.subtraj_len,
                         dim_s=dim_s, dim_z=dim_z,
                         hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        model.init_optimizers()  # TODO
        trainer = BatchHRLTrainer(model, logger, device, env, eval_env=eval_env, tag='imitation',
                                  print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq,
                                  eval_freq=args.eval_freq, eval_num=args.eval_num,
                                  longstep_len=args.subtraj_len)
        batch_preproc = {'observations': (lambda x: x[:, 0, :])}  # use first states only
        trainer.train(demos, num_epochs=args.imitation_task_epochs,
                      batch_size=args.batch_size, num_workers=args.num_workers,
                      batch_preproc=batch_preproc)

    elif args.task_type == 'online':
        from trainers.online_trainer import OnlineHRLTrainer

        # train task policy online
        print('Training task policy ...')
        model = HRLAgent(args.policy_type, opal, longstep_len=args.subtraj_len,
                         dim_s=dim_s, dim_z=dim_z,
                         hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        model.init_optimizers()  # TODO
        trainer = OnlineHRLTrainer(model, logger, device, env, eval_env=eval_env, tag='online',
                                   print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq,
                                   eval_freq=args.eval_freq, eval_num=args.eval_num,
                                   longstep_len=args.subtraj_len)
        trainer.train(train_steps=args.online_train_steps, init_random_steps=args.online_init_random_steps,
                      update_interval=args.online_update_interval, updates_per_step=args.online_updates_per_step,
                      batch_size=args.batch_size)

    elif args.task_type == 'multitask':
        from trainers.online_trainer import OnlineHRLTrainer

        # train task policy online
        print('Training task policy ...')
        model = HRLAgent(args.policy_type, opal, longstep_len=args.subtraj_len, multitask=True,
                         dim_s=dim_s, dim_z=dim_z,
                         hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
        model.init_optimizers()  # TODO
        trainer = OnlineHRLTrainer(model, logger, device, env, eval_env=eval_env, tag='multitask',
                                   print_freq=args.print_freq, log_freq=args.log_freq, save_freq=args.save_freq,
                                   eval_freq=args.eval_freq, eval_num=args.eval_num,
                                   longstep_len=args.subtraj_len)
        trainer.train(train_steps=args.multitask_train_steps, init_random_steps=args.multitask_init_random_steps,
                      update_interval=args.multitask_update_interval, updates_per_step=args.multitask_updates_per_step,
                      batch_size=args.batch_size)

    else:
        raise ValueError('Unknown task type: {}'.format(args.task_type))

    print('#### Task Policy Training End ####')


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


if __name__ == "__main__":
    args = parse_args()
    main(args)
