import time
from typing import Any, Tuple

import draugr
from tqdm import tqdm

from agent.agents.abstract.evo_agent import EVOAgent
from agent.procedures import agent_test_main


class CMAAgent(EVOAgent):

  # region Private

  def __defaults__(self) -> None:
    self._policy = None

  # endregion

  # region Protected

  def _build(self, **kwargs) -> None:
    pass

  def _optimise_wrt(self, error, *args, **kwargs) -> None:
    pass

  def _sample_model(self, state, *args, **kwargs) -> Any:
    pass

  def _train_procedure(self,
                       _environment,
                       rollouts=2000,
                       render=False,
                       render_frequency=100,
                       stat_frequency=10,
                       **kwargs) -> Tuple[Any, Any]:
    training_start_timestamp = time.time()
    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=False)

    stats = draugr.StatisticCollection(stats=('signal', 'duration'))

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        draugr.terminal_plot_stats_shared_x(
            stats,
            printer=E.write,
            )

        E.set_description(f'Epi: {episode_i}, Dur: {stats.duration.running_value[-1]:.1f}')

      if render and episode_i % render_frequency == 0:
        signal, dur, *extras = self.rollout(
            initial_state, _environment, render=render
            )
      else:
        signal, dur, *extras = self.rollout(initial_state, _environment)

      stats.duration.append(dur)
      stats.signal.append(signal)

      if self._end_training:
        break

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(f'\n{"-" * 9} {end_message} {"-" * 9}\n')

    return self._policy, stats

  # endregion

  # region Public

  def sample_action(self, state, *args, **kwargs) -> Any:
    return self._environment.action_space.sample()

  def update(self, *args, **kwargs) -> None:
    pass

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def load(self, *args, **kwargs) -> None:
    pass

  def save(self, *args, **kwargs) -> None:
    pass

  # endregion


# region Test
def cma_test():
  import agent.configs.agent_test_configs.pg_test_config as C

  C.CONNECT_TO_RUNNING = False
  C.ENVIRONMENT_NAME = 'mab'

  agent_test_main(CMAAgent, C)


if __name__ == '__main__':

  cma_test()

# endregion
