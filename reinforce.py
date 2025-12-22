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
    render=False,
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

    optimizer = Adam(params=logits_net.parameters(), lr=lr)

    def train_one_epoch():
        batch_obs = []
        batch_acts = []
        batch_weights = []
        batch_returns = []
        batch_lengths = []

        obs, _ = env.reset()
        done = False

        ep_cum_rew = 0
        ep_len = 0
        while True:
            obs_tensor = torch.tensor(obs)
            action = sample_action(obs_tensor)
            batch_obs.append(obs.copy())

            # returns (obs, rew, done, trunc, info)
            obs, rew, done, _, _ = env.step(action)

            batch_acts.append(action)
            ep_cum_rew += rew
            ep_len += 1

            if done:
                batch_returns.append(ep_cum_rew)
                batch_lengths.append(ep_len)

                batch_weights.extend([ep_cum_rew for _ in range(ep_len)])

                ep_cum_rew = 0
                ep_len = 0
                done = False
                obs, _ = env.reset()

                if len(batch_obs) >= batch_size:
                    break

        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_returns, batch_lengths

    for i in range(epochs):
        batch_loss, batch_returns, batch_lengths = train_one_epoch()
        print(
            (
                f"Epoch: {i + 1}/{epochs}: loss: {float(batch_loss):.3}, "
                f"mean episode return: {float(np.mean(batch_returns)):.3}, "
                f"mean episode length: {float(np.mean(batch_lengths)):.3}"
            )
        )


if __name__ == "__main__":
    train(
        env_name="CartPole-v1",
        hidden_sizes=[32],
        lr=1e-2,
        epochs=50,
        batch_size=5_000,
        render=False,
    )
