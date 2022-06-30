import gym


def uhasd():
    env = gym.make("MineRLNavigateDense-v0")

    obs = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)


def jashd():
    env = gym.make("MineRLNavigateDense-v0")

    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()

        # One can also take a no_op action with
        # action =env.action_space.noop()

        obs, reward, done, info = env.step(action)


def uahsd():
    env = gym.make("MineRLNavigateDense-v0")

    obs = env.reset()
    done = False
    net_reward = 0

    while not done:
        action = env.action_space.noop()

        action["camera"] = [0, 0.03 * obs["compass"]["angle"]]
        action["back"] = 0
        action["forward"] = 1
        action["jump"] = 1
        action["attack"] = 1

        obs, reward, done, info = env.step(action)

        net_reward += reward
        print("Total reward: ", net_reward)


if __name__ == "__main__":
    uhasd()
