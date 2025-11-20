import gymnasium as gym
import torch
import numpy as np

def make_env(env_id="CartPole-v1", seed=0):
    env = gym.make(env_id)
    env.reset(seed=seed)
    return env

def run_random_agent(env_id="CartPole-v1", episodes=5):
    env = make_env(env_id)

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        total_reward = 0.0

        while not (done or truncated):
            action = env.action_space.sample()

            next_obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            obs_t = torch.tensor(obs, dtype=torch.float32)

            obs = next_obs

        print(f"Ep {ep}: reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    run_random_agent(env_id="Taxi-v3")
