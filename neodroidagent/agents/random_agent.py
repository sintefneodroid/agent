from itertools import count
from typing import Any

from tqdm import tqdm

from neodroidagent.interfaces.agent import Agent


class RandomAgent(Agent):

  # region Private

  def __build__(self, env, **kwargs) -> None:
    pass

  # endregion

  # region Public

  def sample(self, state, *args, **kwargs) -> Any:
    return self._last_connected_environment.action_space.sample()

  def update(self, *args, **kwargs) -> None:
    pass

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def rollout(self,
              initial_state,
              environment,
              *,
              train=True,
              render=False,
              **kwargs) -> Any:
    if train:
      self._rollout_i += 1

    episode_signal = 0
    episode_length = 0

    state = initial_state

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False, disable=not render)

    for t in T:
      action = int(self.sample(state)[0])

      state, signal, terminated, info = environment.act(action)
      episode_signal += signal

      if render:
        environment.render()

      if terminated:
        episode_length = t
        break

    if train:
      self.update()

    return episode_signal, episode_length

  def load(self, *args, **kwargs) -> None:
    pass

  def save(self, *args, **kwargs) -> None:
    pass

  # endregion


# region Test
def random_test(rollouts=None, skip=True):
  from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
  from neodroidagent.training.sessions.parallel_training import parallelised_training
  import neodroidagent.configs.agent_test_configs.dqn_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  agent_session_entry_point(RandomAgent,
                            C,
                            parse_args=False,
                            training_session=parallelised_training,
                            skip_confirmation=skip)


if __name__ == '__main__':

  random_test()
# endregion
