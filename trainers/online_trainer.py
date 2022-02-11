import torch

from memory.replay_buffer import ReplayBuffer
from trainers.base_trainer import BaseTrainer


class OnlineTrainer(BaseTrainer):
    def __init__(
        self,
        logger,
        phase,
        tag='',
        print_freq=100,
        log_freq=100,
        save_freq=10,
    ):
        super().__init__(
            logger,
            phase,
            tag,
            print_freq,
            log_freq,
            save_freq,
        )

    def train(
        self,
        model,
        env,
        device,
        longstep_len,
        train_episodes,
        train_start_step,
        updates_per_step,
        batch_size,
        eval_freq,
        eval_episodes,
    ):
        """Off-policy training for high-level policy given fixed low-level policy and prior"""
        replay_buffer = ReplayBuffer(max_size=int(1e6))  # TODO

        # multi-step MDP
        train_start_longstep = (train_start_step + longstep_len - 1) // longstep_len
        total_step, total_longstep = 0, 0

        self.logger.log('phase2_reward_online/train', 0., 0)
        self.logger.log('phase2_reward_online/test', 0., 0)

        # TODO: max total steps?
        for episode in range(1, train_episodes + 1):
            self.reset_timer()
            observation, episode_reward, done = env.reset(), 0., False
            episode_step = 0

            # TODO: max episode length?
            while not done:
                # collect data and put in replay buffer
                with torch.no_grad():
                    if total_longstep < train_start_longstep:  # random exploration
                        # since direct sampling of primitive is impossible, we feed random value to the network
                        primitive = model.get_primitive(env.observation_space.sample())
                    else:
                        primitive = model.get_primitive(observation)

                    longstep_observation = observation
                    longstep_reward = 0.
                    longstep_done = False

                    # accumulate multi-step information
                    for step in range(longstep_len):
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

                    replay_buffer.add((longstep_observation, primitive.cpu().numpy(), longstep_reward,
                                       longstep_next_observation, longstep_done))
                    total_longstep += 1

                # update
                if len(replay_buffer) >= batch_size:
                    for i in range(updates_per_step):
                        samples = replay_buffer.sample(size=batch_size, to_tensor=True, device=device)
                        loss, sublosses = model.update(samples)

                    self.log_losses(total_step, loss, sublosses)  # TODO: total_step might not be divisible by log_freq

            self.logger.log('phase2_reward_online/train', episode_reward, episode)
            print("[phase2_online, episode {}] total steps: {}, episode steps: {}, reward: {:.2f}, mem: {:d}".format(
                episode, total_step, episode_step, episode_reward, len(replay_buffer)))

            # evaluate
            if eval_freq > 0 and episode % eval_freq == 0:
                test_rewards = self.temp_rollout(model, env, eval_episodes, device)
                self.logger.log('phase2_reward_online/test', test_rewards.mean(), episode)
                print("[phase2_online, test] avg reward: {:.2f} (min: {:.2f}, max: {:.2f})".format(
                    test_rewards.mean(), test_rewards.min(), test_rewards.max()))

            self.save_model(model, episode)

        self.save_model(model, train_episodes, force=True)

    # TODO
    def temp_rollout(self, model, env, num_episodes, device):
        import numpy as np
        episode_rewards = []
        for _  in range(num_episodes):
            observation, episode_reward, done = env.reset(), 0., False
            while not done:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
                    action = model.get_action(obs_tensor, deterministic=True).cpu().numpy()
                next_observation, reward, done, _ = env.step(action)
                episode_reward += reward
                observation = next_observation
            episode_rewards.append(episode_reward)
        return np.asarray(episode_rewards)
