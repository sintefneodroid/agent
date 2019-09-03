from itertools import count
from typing import Any

from draugr.writers import Writer, MockWriter
from neodroid.interfaces.unity_specifications import EnvironmentSnapshot

from neodroidagent.interfaces.agent import Agent
from neodroidagent.training.procedures import train_episodically, VectorUnityEnvironment


class RandomAgent(Agent):

  def _update(self, *args, metric_writer: Writer = MockWriter(), **kwargs) -> None:
    pass

  def _sample(self, state: EnvironmentSnapshot, *args, no_random: bool = False,
              metric_writer: Writer = MockWriter(), **kwargs) -> Any:
    self._sample_i += 1
    return self.action_space.sample()

  # region Private

  def __build__(self,observation_space,
                   action_space,
                   signal_space,  **kwargs) -> None:
    self.action_space = action_space

  # endregion

  # region Public

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def rollout(self,
              initial_state,
              environment:VectorUnityEnvironment,
              *,
              train=True,
              render=False,
              **kwargs) -> Any:



    episode_signal = 0
    episode_length = 0

    state = initial_state

    T = count(1)

    for t in T:
      action = self.sample(state)

      state, signal, terminated, info = environment.react(action).to_gym_like_output()

      if terminated[0]:
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


def random_run(rollouts=None, skip=True):
  from neodroidagent.training.agent_session_entry_point import agent_session_entry_point
  from neodroidagent.training.sessions.parallel_training import parallelised_training
  import neodroidagent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  C.CONNECT_TO_RUNNING = True

  agent_session_entry_point(RandomAgent,
                            C,
                            training_session=parallelised_training(training_procedure=train_episodically),
                            skip_confirmation=skip)


if __name__ == '__main__':
  # pg_test()
  random_run()
# endregion

if __name__ == '__main__':

  random_test()
# endregion
