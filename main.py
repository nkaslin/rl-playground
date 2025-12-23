import matplotlib.pyplot as plt
from rl_playground.algorithms.reinforce import train as train_reinforce
from rl_playground.algorithms.reinforce_reward_to_go import train as train_reinforce_reward_to_go
from rl_playground.algorithms.vpg_mc import train as train_vpg_mc
from rl_playground.algorithms.vpg_mc_advantage import train as train_vpg_mc_advantage

# from rl_playground.evaluate.evaluate_algo import ...


# def eval_reinforce():
#     from reinforce import train as train_reinforce
#     from reinforce_with_reward_to_go import (
#         train as train_reinforce_with_reward_to_go,
#     )
#     from reinforce_with_reward_to_go import (
#         train_openai_baseline_implementation as train_baseline,
#     )

#     n_epochs = 50
#     episode_rewards_1 = train_reinforce(
#         env_name="CartPole-v1",
#         hidden_sizes=[32],
#         lr=1e-2,
#         epochs=n_epochs,
#         batch_size=5_000,
#     )

#     episode_rewards_2 = train_reinforce_with_reward_to_go(
#         env_name="CartPole-v1",
#         hidden_sizes=[32],
#         lr=1e-2,
#         epochs=n_epochs,
#         batch_size=5_000,
#     )

#     episode_rewards_3 = train_baseline(
#         env_name="CartPole-v1",
#         hidden_sizes=[32],
#         lr=1e-2,
#         epochs=n_epochs,
#         batch_size=5_000,
#     )

#     eps = list(range(1, n_epochs + 1))
#     plt.plot(
#         eps,
#         episode_rewards_1,
#         c="r",
#         label="reinforce",
#     )
#     plt.plot(
#         eps,
#         episode_rewards_2,
#         c="g",
#         label="reinforce r2g",
#     )
#     plt.plot(
#         eps,
#         episode_rewards_3,
#         c="b",
#         label="reinforce r2g openai",
#     )
#     plt.xlabel("epoch")
#     plt.ylabel("reward")
#     plt.legend()
#     plt.savefig("reinforce_comparison.png")


def main():
    # eval_reinforce()
    # TODO: implement proper eval of different algos on cartpole
    ...


if __name__ == "__main__":
    main()
