from itertools import count

from tqdm import tqdm

from neodroid.environments.unity_environment import VectorUnityEnvironment


def run(self, environment: VectorUnityEnvironment, render: bool = True) -> None:
    state = environment.reset().observables

    F = count(1)
    F = tqdm(F, leave=False, disable=not render)
    for frame_i in F:
        F.set_description(f"Frame {frame_i}")

        action, *_ = self.sample(state, deterministic=True)
        state, signal, terminated, info = environment.react(action, render=render)

        if terminated.all():
            state = environment.reset().observables


def infer(self, env, render=True):
    for episode_i in count(1):
        print(f"Episode {episode_i}")
        state = env.reset()

        for frame_i in count(1):

            action, *_ = self.sample(state)
            state, signal, terminated, info = env.act(action)
            if render:
                env.render()

            if terminated:
                break
