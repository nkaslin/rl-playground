"""
Some implementation details:

1) Rollout: collect trajectories with frozen policy
2) Compute the values for the encountered observations
3) Compute the residuals using the computed value estimates and 
    the observed rewards. 
    res_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    with V(s_{t+1}) = 0 at end of episode (where done == True)
4) Backward computation of advantages
    A_t = delta_t + gamma * lambda A_{t+1}
    with A_{t+1} = 0 at end of episode
5) update policy
6) use the value estimates and the advantage estimates to get the value targets
    and regress the value function on the MSE loss
"""


import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from rl_playground.nn.mlp import mlp


def train(
    env_name="CartPole-v1",
    policy_hidden_sizes=[32],
    policy_lr=1e-2,
    epochs=50,
    batch_size=5000,
    gamma=0.99,
    lambda_=1.0,
    normalize_adv=True,
    value_fcn_hidden_sizes=[32],
    alpha=5e-3,
    value_iterations_per_epoch=5,
):
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), (
        "This example only works for envs with continuous state spaces."
    )
    assert isinstance(env.action_space, Discrete), (
        "This example only works for envs with discrete action spaces."
    )

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    logits_net = mlp(sizes=[obs_dim] + policy_hidden_sizes + [n_acts])

    value_net = mlp(
        sizes=[obs_dim] + value_fcn_hidden_sizes + [1],
    )

    def get_policy(obs):
        return Categorical(logits=logits_net(obs))

    def sample_action(obs):
        return get_policy(obs).sample().item()

    def compute_loss(obs, act, adv):
        logp = get_policy(obs).log_prob(act)
        return -(logp * adv).mean()

    def compute_residuals(values, rewards, dones):
        residuals = []
        val_at_plus_one = 0.0
        for val, rew, done in reversed(list(zip(values, rewards, dones))):
            delta = rew + gamma * val_at_plus_one - val
            residuals.append(delta)
            val_at_plus_one = val if not done else 0.0
        return list(reversed(residuals))
    
    def compute_advantages(residuals, dones):
        advantages = []
        adv_at_plus_one = 0.0
        for res, done in reversed(list(zip(residuals, dones))):
            adv = res + gamma * lambda_ * adv_at_plus_one
            advantages.append(adv)
            adv_at_plus_one = adv if not done else 0.0
        return list(reversed(advantages))

    def compute_value_estimates(episode_obs):
        obs_tensor = torch.tensor(np.array(episode_obs), dtype=torch.float32)
        return value_net(obs_tensor).squeeze()

    policy_optimizer = Adam(params=logits_net.parameters(), lr=policy_lr)
    value_fcn_optimizer = Adam(params=value_net.parameters(), lr=alpha)

    def train_one_epoch():
        def value_fct_training_iteration(returns, values):
            value_fcn_optimizer.zero_grad()
            loss = ((values - returns) ** 2).mean()
            loss.backward()
            value_fcn_optimizer.step()

        # 1) rollout
        obs_t = []
        act_t = []
        reward_t = []
        done_t = []

        ep_lengths, ep_rewards = [], []

        obs, _ = env.reset()
        done = False

        cur_len, cur_rew = 0, 0.0
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = sample_action(obs_tensor)
            obs_t.append(obs.copy())

            # returns (obs, rew, done, trunc, info)
            obs, rew, done, _, _ = env.step(action)

            act_t.append(action)
            reward_t.append(rew)
            done_t.append(done)

            cur_len += 1
            cur_rew += rew

            if done:
                done = False
                obs, _ = env.reset()

                ep_lengths.append(cur_len)
                ep_rewards.append(cur_rew)
                cur_len, cur_rew = 0, 0.0

                if len(obs_t) >= batch_size:
                    break

        # 2) compute value estimates (does not require grad)
        with torch.no_grad():
            value_estimates_t = compute_value_estimates(obs_t)

        # 3) compute residuals
        residuals_t = compute_residuals(value_estimates_t, reward_t, done_t)

        # 4) compute advantage estimates 
        adv = compute_advantages(residuals_t, done_t)
        adv = torch.tensor(adv, dtype=torch.float32)

        # normalized adv usually preforms better
        if normalize_adv:
            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 5) policy update
        policy_optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(obs_t, dtype=torch.float32),
            act=torch.as_tensor(act_t, dtype=torch.int32),
            adv=adv_norm,
        )
        batch_loss.backward()
        policy_optimizer.step()

        # 6) value function update loop (multiple iterations)
        # TODO: (possible extension) add minibatch functionality to value function update
        value_targets = adv + value_estimates_t
        obs_tensor = torch.as_tensor(obs_t, dtype=torch.float32)
        for _ in range(value_iterations_per_epoch):
            current_values = value_net(obs_tensor).squeeze(-1)
            value_fct_training_iteration(value_targets, current_values)

        return batch_loss, ep_rewards, ep_lengths

    batch_returns = []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lengths = train_one_epoch()
        batch_returns.append(np.mean(batch_rets))
        print(
            "epoch: %3d \t loss: %.2f \t return: %.2f \t ep_len: %.2f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lengths))
        )

    return batch_returns


if __name__ == "__main__":
    train(
        env_name="CartPole-v1",
        policy_hidden_sizes=[32],
        policy_lr=1e-2,
        epochs=80,
        batch_size=5_000,
        gamma=0.99,
        lambda_=1.0,
        normalize_adv=True,
        value_fcn_hidden_sizes=[32],
        alpha=5e-3,
        value_iterations_per_epoch=5,
    )
