import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from rl_playground.nn.mlp import mlp


def train(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
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
                cum_ep_rewards = np.cumsum(episode_rewards)
                total_episode_reward = cum_ep_rewards[-1]
                batch_returns.append(total_episode_reward)
                batch_lengths.append(len(episode_rewards))

                batch_weights.extend(
                    [
                        total_episode_reward - cum_ep_rewards[i]
                        for i in range(len(episode_rewards))
                    ]
                )

                episode_rewards = []
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

    batch_returns = []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lengths = train_one_epoch()
        batch_returns.append(np.mean(batch_rets))
        print(
            "epoch: %3d \t loss: %.2f \t return: %.2f \t ep_len: %.2f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lengths))
        )

    return batch_returns


# openai implementation to test my implementation against:
# https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/2_rtg_pg.py
def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i + 1] if i + 1 < n else 0)
    return rtgs


def train_openai_baseline_implementation(
    env_name="CartPole-v1",
    hidden_sizes=[32],
    lr=1e-2,
    epochs=50,
    batch_size=5000,
    render=False,
):
    # make environment, check spaces, get obs / act dims
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

    # make function to compute action distribution
    def get_policy(obs):
        logits = logits_net(obs)
        return Categorical(logits=logits)

    # make action selection function (outputs int actions, sampled from policy)
    def get_action(obs):
        return get_policy(obs).sample().item()

    # make loss function whose gradient, for the right data, is policy gradient
    def compute_loss(obs, act, weights):
        logp = get_policy(obs).log_prob(act)
        return -(logp * weights).mean()

    # make optimizer
    optimizer = Adam(logits_net.parameters(), lr=lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []  # for observations
        batch_acts = []  # for actions
        batch_weights = []  # for reward-to-go weighting in policy gradient
        batch_rets = []  # for measuring episode returns
        batch_lens = []  # for measuring episode lengths

        # reset episode-specific variables
        obs, _ = env.reset()  # first obs comes from starting distribution
        done = False  # signal from environment that episode is over
        ep_rews = []  # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:
            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            act = get_action(torch.as_tensor(obs, dtype=torch.float32))
            obs, rew, done, _, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a_t|s_t) is reward-to-go from t
                batch_weights += list(reward_to_go(ep_rews))

                # reset episode-specific variables
                obs, _ = env.reset()
                done, ep_rews = False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss = compute_loss(
            obs=torch.as_tensor(batch_obs, dtype=torch.float32),
            act=torch.as_tensor(batch_acts, dtype=torch.int32),
            weights=torch.as_tensor(batch_weights, dtype=torch.float32),
        )
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens


    # training loop
    batch_returns = []
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        batch_returns.append(np.mean(batch_rets))
        print(
            "epoch: %3d \t loss: %.2f \t return: %.2f \t ep_len: %.2f"
            % (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens))
        )

    return batch_returns

if __name__ == "__main__":
    train(
        env_name="CartPole-v1",
        hidden_sizes=[32],
        lr=1e-2,
        epochs=50,
        batch_size=5_000,
    )
    # train_openai_baseline_implementation(
    #     env_name="CartPole-v1",
    #     hidden_sizes=[32],
    #     lr=1e-2,
    #     epochs=50,
    #     batch_size=5_000,
    # )
