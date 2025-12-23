import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from nn import mlp


def train(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    gamma=0.99,
    normalize_adv=True,
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
    logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

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

    optimizer = Adam(params=logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_advantages = []
        batch_returns = []
        batch_lengths = []

        obs, _ = env.reset()
        done = False

        episode_rewards = []
        while True:
            obs_tensor = torch.tensor(obs)
            action = sample_action(obs_tensor)
            batch_obs.append(obs.copy())

            # returns (obs, rew, done, trunc, info)
            obs, rew, done, _, _ = env.step(action)

            batch_acts.append(action)
            episode_rewards.append(rew)

            if done:
                adv = compute_discounted_return(episode_rewards)

                batch_returns.append(sum(episode_rewards))
                batch_lengths.append(len(episode_rewards))

                batch_advantages.extend(adv)

                episode_rewards = []
                done = False
                obs, _ = env.reset()

                if len(batch_obs) >= batch_size:
                    break

        adv = torch.as_tensor(batch_advantages, dtype=torch.float32)

        # normalized adv usually preforms better
        if normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_advantages, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
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
        hidden_sizes=[32],
        lr=1e-2,
        epochs=50,
        batch_size=5_000,
        gamma=0.99,
        normalize_adv=True,
    )
