import time
from itertools import count
from typing import Any, Tuple

from tqdm import tqdm

import utilities as U
from agents.abstract.agent import Agent


class RandomAgent(Agent):
  def _build(self) -> None:
    pass

  def _defaults(self) -> None:
    self._policy = None

  def sample_action(self, state, *args, **kwargs) -> Any:
    return self._environment.action_space.sample()

  def _sample_model(self, state, *args, **kwargs) -> Any:
    pass

  def update(self, *args, **kwargs) -> None:
    pass

  def _optimise_wrt(self, error, *args, **kwargs) -> None:
    pass

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def rollout(self, initial_state, environment, *, train=True, render=False, **kwargs) -> Any:
    if train:
      self._rollout_i += 1

    episode_signal = 0
    episode_length = 0

    state = initial_state

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      action = self.sample_action(state)

      state, signal, terminated, info = environment.step(action=action)
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

  def _train(self,
             _environment,
             rollouts=2000,
             render=False,
             render_frequency=100,
             stat_frequency=10,
             **kwargs) -> Tuple[Any, Any]:
    training_start_timestamp = time.time()
    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=False)

    stats = U.StatisticCollection(stats=('signal', 'duration'))

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        U.term_plot_stats_shared_x(
            stats,
            printer=E.write,
            )

        E.set_description(f'Episode: {episode_i}, Running length: {stats.duration.running_value[-1]}')

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
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s'
    print('\n{} {} {}\n'.format('-' * 9, end_message, '-' * 9))

    return self._policy, stats


if __name__ == '__main__':
  import configs.agent_test_configs.test_pg_config as C

  C.CONNECT_TO_RUNNING = False
  C.ENVIRONMENT_NAME = 'grd'

  U.test_agent_main(RandomAgent, C)
