"""
The discounted reward-to-go G_t is the Monte-Carlo estimate of Q(s_t, a_t), therefore
in combination with a value estimate, we have an advantage estimate based on the Monte-
Carlo estimate of Q.

A_t = G_t - V(s_t)
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

    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    def compute_discounted_return(episode_rewards):
        G = 0
        returns = []
        for r in reversed(episode_rewards):
            G = r + gamma * G
            returns.append(G)
        return list(reversed(returns))

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

        batch_obs = []
        batch_acts = []
        batch_discounted_rewards_to_go = []
        batch_returns = []
        batch_lengths = []

        obs, _ = env.reset()
        done = False

        episode_rewards = []
        episode_obs = []
        while True:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            action = sample_action(obs_tensor)
            episode_obs.append(obs.copy())

            # returns (obs, rew, done, trunc, info)
            obs, rew, done, _, _ = env.step(action)

            batch_acts.append(action)
            episode_rewards.append(rew)

            if done:
                batch_discounted_rewards_to_go.extend(
                    compute_discounted_return(episode_rewards)
                )

                batch_returns.append(sum(episode_rewards))
                batch_lengths.append(len(episode_rewards))
                batch_obs.extend(episode_obs)

                episode_rewards = []
                episode_obs = []
                done = False
                obs, _ = env.reset()

                if len(batch_obs) >= batch_size:
                    break

        batch_discounted_rewards_to_go = torch.as_tensor(
            batch_discounted_rewards_to_go, dtype=torch.float32
        )

        with torch.no_grad():
            batch_value_estimates = compute_value_estimates(batch_obs)
        adv = batch_discounted_rewards_to_go - batch_value_estimates

        # normalized adv usually preforms better
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=adv,
        )
        batch_loss.backward()
        policy_optimizer.step()

        # value function update loop
        # TODO: (possible extension) add minibatch functionality to value function update
        obs_tensor = torch.as_tensor(batch_obs, dtype=torch.float32)
        for _ in range(value_iterations_per_epoch):
            current_values = value_net(obs_tensor).squeeze(-1)
            value_fct_training_iteration(batch_discounted_rewards_to_go, current_values)

        return batch_loss, batch_returns, batch_lengths

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
        normalize_adv=True,
        value_fcn_hidden_sizes=[32],
        alpha=5e-3,
        value_iterations_per_epoch=5,
    )
