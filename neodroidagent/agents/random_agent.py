from itertools import count
from typing import Any

from draugr import MockWriter, Writer
from neodroid.environments.unity_environment import VectorUnityEnvironment
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroid.utilities.unity_specifications import EnvironmentSnapshot
from neodroidagent.agents.agent import Agent
from neodroidagent.procedures import StepWise


class RandomAgent(Agent):

  def _update(self,
              *args,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> None:
    pass

  def _sample(self,
              state: EnvironmentSnapshot,
              *args,
              no_random: bool = False,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> Any:
    self._sample_i += 1
    return self.action_space.sample()

  # region Private

  def __build__(self,
                observation_space: ObservationSpace,
                action_space: ActionSpace,
                signal_space: SignalSpace,
                **kwargs) -> None:
    self.action_space = action_space

  # endregion

  # region Public

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def rollout(self,
              initial_state,
              environment: VectorUnityEnvironment,
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
  from neodroidagent.sessions.session_entry_point import session_entry_point
  from neodroidagent.sessions.single_agent.parallel import ParallelSession
  import neodroidagent.configs.agent_test_configs.dqn_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  session_entry_point(RandomAgent,
                      C,
                      parse_args=False,
                      session=ParallelSession,
                      skip_confirmation=skip)


def random_run(rollouts=None, skip=True):
  from neodroidagent.sessions.session_entry_point import session_entry_point
  from neodroidagent.sessions.single_agent.parallel import ParallelSession
  import neodroidagent.configs.agent_test_configs.pg_test_config as C

  if rollouts:
    C.ROLLOUTS = rollouts

  session_entry_point(RandomAgent,
                      C,
                      session=ParallelSession(StepWise,
                                              auto_reset_on_terminal_state=True,
                                              connect_to_running=True),
                      skip_confirmation=skip)


if __name__ == '__main__':
  # pg_test()
  random_run()
# endregion

if __name__ == '__main__':

  random_test()
# endregion
