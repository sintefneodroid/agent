import time
from itertools import count
from typing import Any, Tuple

from tqdm import tqdm

import draugr
from agent.interfaces.partials.agents.torch_agents.torch_agent import TorchAgent


class RandomAgent(TorchAgent):

  # region Private

  def __defaults__(self) -> None:
    self._policy = None

  # endregion

  # region Protected

  def _build(self, **kwargs) -> None:
    pass

  def _optimise_wrt(self, error, **kwargs) -> None:
    pass

  def _sample_model(self, state, **kwargs) -> Any:
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
    E = tqdm(E, f'Episode: {1}', leave=False, disable=not render)

    stats = draugr.StatisticCollection(stats=('signal', 'duration'))

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        draugr.terminal_plot_stats_shared_x(stats,
                                            printer=E.write,
                                            )

        E.set_description(f'Epi: {episode_i}, Dur: {stats.duration.running_value[-1]:.1f}')

      if render and episode_i % render_frequency == 0:
        signal, dur, *extras = self.rollout(initial_state,
                                            _environment,
                                            render=render
                                            )
      else:
        signal, dur, *extras = self.rollout(initial_state, _environment)

      stats.duration.append(dur)
      stats.signal.append(signal)

      if self.end_training:
        break

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(f'\n{"-" * 9} {end_message} {"-" * 9}\n')

    return self._policy, stats

  # endregion

  # region Public

  def sample_action(self, state, *args, **kwargs) -> Any:
    return self._environment.action_space._sample()

  def update(self, *args, **kwargs) -> None:
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
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False, disable=not render)

    for t in T:
      action = int(self.sample_action(state)[0])

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
def random_test():
  import neodroid.wrappers.gym_wrapper as neo
  env = neo.NeodroidGymWrapper(environment_name='mab')
  agent = RandomAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      environment=env)
  agent.build(env)
  agent.train(env, env)


if __name__ == '__main__':

  random_test()
# endregion
