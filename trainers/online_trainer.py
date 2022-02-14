import torch

from memory.replay_buffer import ReplayBuffer
from trainers.base_trainer import BaseTrainer


class OnlineTrainer(BaseTrainer):
    def __init__(
        self,
        logger,
        tag='',
        on_policy=False,
        print_freq=100,
        log_freq=100,
        save_freq=10,
    ):
        super().__init__(
            logger,
            tag,
            print_freq,
            log_freq,
            save_freq,
        )
        self.on_policy = on_policy

    def train(
        self,
        model,
        env,
        device,
        longstep_len,
        train_steps=int(2.5e6),
        init_random_steps=10000,
        update_interval=1,
        updates_per_step=1,
        batch_size=256,
        eval_freq=10,
        eval_episodes=5,
        max_buffer_size=int(1e6),
        multitask=False,
    ):
        """Online training for high-level policy given fixed low-level policy and prior"""
        replay_buffer = ReplayBuffer(max_size=max_buffer_size)
        if self.on_policy:
            batch_size = update_interval

        # multi-step MDP
        init_random_longsteps = (init_random_steps + longstep_len - 1) // longstep_len
        total_step, total_longstep = 0, 0

        self.logger.log('{}_reward/train'.format(self.tag), 0., 0)
        self.logger.log('{}_reward/test'.format(self.tag), 0., 0)

        episode = 0
        while total_step < train_steps:
            self.reset_timer()
            observation, episode_reward, done = env.reset(), 0., False
            episode += 1
            episode_step = 0

            while not done and total_step < train_steps:
                # collect data and put in replay buffer
                with torch.no_grad():
                    if total_longstep < init_random_longsteps:  # random exploration
                        # since direct sampling of primitive is impossible, we feed random value to the network
                        random_obs = env.observation_space.sample()
                        primitive_and_info = model.get_primitive(random_obs, return_logprob=self.on_policy)
                    else:
                        primitive_and_info = model.get_primitive(observation, return_logprob=self.on_policy)

                    if isinstance(primitive_and_info, tuple):
                        primitive, extra_info = primitive_and_info[0], primitive_and_info[1:]
                    else:
                        primitive, extra_info = primitive_and_info, tuple()

                    longstep_observation = observation
                    longstep_reward = 0.
                    longstep_done = False

                    # accumulate multi-step information
                    if multitask:
                        # get multiple actions at once
                        observation_dup = observation[None, :].repeat(longstep_len, axis=0)
                        primitive_dup = primitive.unsqueeze(0).repeat(longstep_len, 1)
                        actions = model.get_action_from_primitive(observation_dup, primitive_dup).cpu().numpy()

                    for step in range(longstep_len):
                        if multitask:
                            action = actions[step]
                        else:
                            action = model.get_action_from_primitive(observation, primitive).cpu().numpy()
                        action = action.clip(env.action_space.low, env.action_space.high)

                        next_observation, reward, done, _ = env.step(action)
                        longstep_reward += reward

                        episode_reward += reward
                        episode_step += 1
                        total_step += 1

                        # Ignore the "done" signal if it comes from hitting the time horizon
                        # https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/sac/sac.py
                        true_done = False if episode_step == env.spec.max_episode_steps else done
                        longstep_done |= true_done

                        observation = next_observation
                        longstep_next_observation = next_observation

                        if done:
                            break

                # TODO: handle early stopping
                replay_buffer.add((longstep_observation, primitive.cpu().numpy(), longstep_reward,
                                   longstep_next_observation, longstep_done) + extra_info)
                total_longstep += 1

                # update
                if total_longstep % update_interval == 0 and len(replay_buffer) >= batch_size:
                    for i in range(updates_per_step):
                        samples = replay_buffer.sample(size=batch_size, to_tensor=True, device=device)
                        loss, sublosses = model.update(samples)

                    if self.on_policy:
                        replay_buffer.clear()

                    log_step = (total_step // longstep_len) * longstep_len
                    self.log_losses(log_step, loss, sublosses)

                # adjust parameters
                if hasattr(model, 'adjust_params'):
                    model.adjust_params(total_step)  # TODO: longstep?

            self.logger.log('{}_reward/train'.format(self.tag), episode_reward, episode)
            print("[{}, episode {}] total steps: {}, episode steps: {}, reward: {:.2f}, mem: {:d}".format(
                self.tag, episode, total_step, episode_step, episode_reward, len(replay_buffer)))

            # evaluate
            if eval_freq > 0 and episode % eval_freq == 0:
                if multitask:
                    test_rewards = self.rollout_multistep(model, env, eval_episodes, longstep_len)
                else:
                    test_rewards = self.rollout(model, env, eval_episodes)
                self.logger.log('{}_reward/test'.format(self.tag), test_rewards.mean(), episode)
                print("[{}, test] avg reward: {:.2f} (min: {:.2f}, max: {:.2f})".format(
                    self.tag, test_rewards.mean(), test_rewards.min(), test_rewards.max()))

            self.save_model(model, episode)

        self.save_model(model, episode, force=True)

    def rollout(self, model, env, num_episodes):
        import numpy as np
        episode_rewards = []
        for _  in range(num_episodes):
            observation, episode_reward, done = env.reset(), 0., False
            while not done:
                with torch.no_grad():
                    action = model.get_action(observation, deterministic=True).cpu().numpy()
                action = np.clip(action, env.action_space.low, env.action_space.high)
                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                observation = next_observation
            episode_rewards.append(episode_reward)
        return np.asarray(episode_rewards)

    def rollout_multistep(self, model, env, num_episodes, longstep_len):
        import numpy as np
        episode_rewards = []
        for _  in range(num_episodes):
            observation, episode_reward, done = env.reset(), 0., False
            while not done:
                with torch.no_grad():
                    primitive = model.get_primitive(observation)
                    observation_dup = observation[None, :].repeat(longstep_len, axis=0)
                    primitive_dup = primitive.unsqueeze(0).repeat(longstep_len, 1)
                    actions = model.get_action_from_primitive(observation_dup, primitive_dup, deterministic=True).cpu().numpy()

                for step in range(longstep_len):
                    action = actions[step]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    next_observation, reward, done, _ = env.step(action)
                    episode_reward += reward
                    observation = next_observation
                    if done:
                        break
            episode_rewards.append(episode_reward)
        return np.asarray(episode_rewards)
