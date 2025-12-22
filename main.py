from reinforce import train as train_reinforce


def main():
    train_reinforce(
        env_name="CartPole-v1",
        hidden_sizes=[32],
        lr=8e-3,
        epochs=50,
        batch_size=10_000,
        render=False,
    )


if __name__ == "__main__":
    main()
