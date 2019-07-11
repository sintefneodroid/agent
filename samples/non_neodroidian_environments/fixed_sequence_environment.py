import numpy as np
from gym.spaces import Discrete

from samples.non_neodroidian_environments.constant_environment import NoRenderEnv


class FixedSequenceEnvironment(NoRenderEnv):

  def __init__(self,
               n_actions=10,
               seed=0,
               episode_length=100
               ):
    self._episode_len = episode_length

    self.np_random = np.random.RandomState()
    self.np_random.seed(seed)
    self.sequence = [self.np_random.randint(0, n_actions - 1) for _ in range(episode_length)]

    self.action_space = Discrete(n_actions)
    self.observation_space = Discrete(1)

    self.reset()

  def reset(self):
    self._time = 0
    return 0

  def step(self, actions):
    signal = self._get_signal(actions)
    self._to_next_state()
    if self._episode_len and self.time >= self._episode_len:
      return 0, 0, True, {}

    return 0, signal, False, {}

  @property
  def time(self):
    return self._time

  def _to_next_state(self):
    self._time += 1

  def _get_signal(self, actions):
    return 1 if actions == self.sequence[self.time] else 0


if __name__ == '__main__':
  env = FixedSequenceEnvironment()
  obs = env.reset()
  for t in range(100000):
    obs, signal, term, info = env.step(env.np_random.randint(0, env.action_space.n - 1))
    print(obs, signal, term)
    if term:
      obs = env.reset()
