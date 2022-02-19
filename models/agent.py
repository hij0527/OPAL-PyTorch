import numpy as np
import torch
from torch.distributions import Normal


class RLAgent:
    def __init__(self, policy_type, **policy_kwargs):
        self.policy_type = policy_type
        self.policy = self._get_policy(policy_type, **policy_kwargs)
        self.on_policy = policy_type in ['ppo']

    def to(self, device):
        self.device = device
        self.policy.to(device)
        return self

    def init_optimizers(self, **lrs):
        self.policy.init_optimizers(**lrs)

    def update(self, buffer, batch_size, num_updates, **kwargs):
        return self.policy.update(buffer, batch_size, num_updates, **kwargs)

    def get_action(self, state, deterministic=True):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        mean, logstd = self.policy(state.unsqueeze(0))
        mean, logstd = mean.squeeze(0), logstd.squeeze(0)

        if deterministic:
            action = mean
            logprob = torch.zeros_like(action, device=self.device).sum(-1)
        else:
            dist = Normal(mean, logstd.exp())
            action = dist.rsample()
            logprob = dist.log_prob(action).sum(-1)

        return action, logprob

    @torch.no_grad()
    def rollout_step(self, env, episode_step, observation, rand=False, deterministic=False):
        if rand:
            action = env.action_space.sample()
            logprob = np.zeros_like(action).sum(-1)
        else:
            action, logprob = self.get_action(observation, deterministic=deterministic)
            action, logprob = action.cpu().numpy(), logprob.cpu().numpy()
        action = np.clip(action, env.action_space.low, env.action_space.high)
        next_observation, reward, done, _ = env.step(action)
        episode_step += 1
        success = False if episode_step == env.spec.max_episode_steps else done
        return episode_step, next_observation, reward, done, success, action, logprob

    def state_dict(self):
        return {
            'policy': self.policy.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.policy.load_state_dict(state_dict['policy'])

    def _get_policy(self, policy_type, **policy_kwargs):
        if policy_type == 'sac':
            from models.policies.sac import SAC
            policy_cls = SAC
        elif policy_type == 'ppo':
            from models.policies.ppo import PPO
            policy_cls = PPO
            policy_kwargs.update({
                'activation': 'relu',
            })
        elif policy_type == 'bc':
            from models.policies.bc import BC
            policy_cls = BC
        elif policy_type == 'cql':
            from models.policies.cql import CQL
            policy_cls = CQL
            param_str = policy_kwargs.pop('policy_param_str')
            reg_type, alpha, tau = param_str.split('_')
            policy_kwargs.update({
                'reg_type': 'H' if reg_type == 'h' else reg_type,
                'alpha_prime': float(alpha),
                'alpha_prime_min': 0.001,
                'lagrange_thresh': float(tau),
                'with_lagrange': alpha == 0,
            })
        else:
            raise ValueError('Unsupported policy type: {}'.format(policy_type))

        return policy_cls(**policy_kwargs)


class HRLAgent(RLAgent):
    def __init__(self, policy_type, opal, longstep_len=1, multitask=False, **policy_kwargs):
        super().__init__(policy_type, **policy_kwargs)
        self.opal = opal
        self.longstep_len = longstep_len
        self.multitask = multitask

    def to(self, device):
        super().to(device)
        self.opal.to(device)
        return self

    def update(self, buffer, batch_size, num_updates, **kwargs):
        if self.policy_type == 'sac':
            kwargs.update({'prior_model': self.opal.prior})
        return super().update(buffer, batch_size, num_updates, **kwargs)

    def get_primitive(self, state, deterministic=False):
        return super().get_action(state, deterministic)

    def get_action_from_primitive(self, state, latent, deterministic=False):
        return self.opal.get_action(state, latent, deterministic)

    def get_action(self, state, deterministic=True):
        # TODO: this doesn't work for multi-task agent
        latent, latent_logprob = self.get_primitive(state, deterministic)
        action, action_logprob = self.get_action_from_primitive(state, latent, deterministic)
        logprob = latent_logprob + action_logprob
        return action, logprob

    @torch.no_grad()
    def rollout_step(self, env, episode_step, observation, rand=False, deterministic=False):
        if rand:
            # since direct sampling of primitive is impossible, we feed random value to the network
            random_obs = env.observation_space.sample()
            primitive, primitive_logprob = self.get_primitive(random_obs, deterministic=deterministic)
        else:
            primitive, primitive_logprob = self.get_primitive(observation, deterministic=deterministic)

        longstep_reward = 0.
        success = False

        # accumulate multi-step information
        if self.multitask:
            # get multiple actions at once
            observation_dup = observation[None].repeat(self.longstep_len, axis=0)
            primitive_dup = primitive[None].repeat(self.longstep_len, 1)
            actions, _ = self.get_action_from_primitive(observation_dup, primitive_dup, deterministic=deterministic)

        for step in range(self.longstep_len):
            if self.multitask:
                action = actions[step]
            else:
                action, _ = self.get_action_from_primitive(observation, primitive, deterministic=deterministic)
            action = action.cpu().numpy()
            action = action.clip(env.action_space.low, env.action_space.high)

            next_observation, reward, done, _ = env.step(action)
            longstep_reward += reward
            episode_step += 1
            success |= False if episode_step == env.spec.max_episode_steps else done
            observation = next_observation

            if done:
                break

        return episode_step, next_observation, longstep_reward, done, success, \
            primitive.cpu().numpy(), primitive_logprob.cpu().numpy()

    @torch.no_grad()
    def rollout_single_step(self, env, episode_step, observation, rand=False, deterministic=False):
        return super().rollout_step(env, episode_step, observation, rand, deterministic)

    def _get_policy(self, policy_type, **policy_kwargs):
        policy_kwargs['dim_a'] = policy_kwargs.pop('dim_z')
        if policy_type == 'sac':
            from models.policies.sac import SAC
            policy_cls = SAC
            policy_kwargs.update({'use_kld_penalty': True})
        elif policy_type == 'cql':
            from models.policies.cql import CQL
            policy_cls = CQL
            param_str = policy_kwargs.pop('policy_param_str')
            reg_type, alpha, tau = param_str.split('_')
            policy_kwargs.update({
                'reg_type': 'H' if reg_type == 'h' else reg_type,
                'alpha_prime': float(alpha),
                'alpha_prime_min': 0.001,
                'lagrange_thresh': float(tau),
                'with_lagrange': alpha == 0,
                'adjust_action_range': True,
            })
        else:
            return super()._get_policy(policy_type, **policy_kwargs)

        return policy_cls(**policy_kwargs)
