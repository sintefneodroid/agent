import typing
from collections import defaultdict
from itertools import count
from typing import Any, Tuple

import gym
import numpy as np
from tqdm import tqdm

from agent.interfaces.partials.agents.torch_agents.value_agent import ValueAgent


class TabularQAgent(ValueAgent):
  '''
  Agent implementing tabular Q-learning.
  '''

  # region Private

  def __defaults__(self) -> None:

    self._action_n = 6

    self._init_mean = 0.0
    self._init_std = 0.1
    self._learning_rate = .6
    self._discount_factor = .95

    self._initial_observation_period = 0

  # endregion

  # region Public

  def update(self, *args, **kwargs) -> None:
    pass

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def load(self, *args, **kwargs) -> None:
    pass

  def save(self, *args, **kwargs) -> None:
    pass

  def sample_action(self, state, **kwargs):
    if not isinstance(state, str):
      state = str(state)

    return super().sample_action(state)

  def sample_random_process(self):
    if hasattr(self._environment.action_space, 'signed_one_hot_sample'):
      return self._environment.action_space.signed_one_hot_sample()
    else:
      return self._environment.action_space._sample()

  def rollout(self,
              initial_state,
              environment,
              *,
              train=True,
              render=False,
              **kwargs) -> Any:
    obs = initial_state
    ep_r = 0
    steps = 0
    for t in count():
      action = self.sample_action(obs)

      next_obs, signal, terminal, _ = environment.act(action)

      next_obs = str(next_obs)

      current_q = self._q_table[obs, action]
      future = np.max(self._q_table[next_obs])
      exp_q = signal + self._discount_factor * future
      diff = self._learning_rate * (exp_q - current_q)
      self._q_table[obs, action] = current_q + diff
      #        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])

      obs = next_obs
      ep_r += signal

      if terminal:
        print(signal)
        steps = t
        break

    return ep_r, steps

  def train_episodically(
      self,
      env,
      *,
      rollouts=1000,
      render=False,
      render_frequency=100,
      stat_frequency=10,
      **kwargs
      ):
    obs = env.reset()
    obs = str(obs)

    for i in range(rollouts):
      episode_signal, steps = self.rollout(obs, env)
      obs = env.reset()

    return self._q_table

  # endregion

  # region Protected

  def _sample_model(self,
                    state,
                    *args,
                    **kwargs) -> Any:
    if isinstance(state, typing.Collection) and len(state) == 0:
      return [0]
    return np.argmax(self._q_table[state])

  def _optimise_wrt(self,
                    error,
                    *args,
                    **kwargs) -> None:
    pass

  def _build(self, **kwargs) -> None:

    if hasattr(self._environment.action_space, 'num_binary_actions'):
      self._action_n = self._environment.action_space.num_binary_actions
    else:
      self._action_n = self._environment.action_space.n

    # self._verbose = True

    self._q_table = defaultdict(
        lambda:self._init_std * np.random.randn(self._action_n) + self._init_mean)

    # self._q_table = np.zeros([self._environment.observation_space.n, self._environment.action_space.n])

  def _train_procedure(self,
                       env,
                       rollouts=10000,
                       *args,
                       **kwargs) -> Tuple[Any, Any]:
    model, *stats = self.train_episodically(env, rollouts=rollouts)

    return model, stats

  # endregion


# region Test
def tabular_test():
  import neodroid.wrappers.gym_wrapper as neo
  env = neo.NeodroidGymWrapper(environment_name='mab')
  agent = TabularQAgent(observation_space=env.observation_space,
                        action_space=env.action_space,
                        environment=env)
  agent.build(env)
  agent.train(env, env)


if __name__ == '__main__':
  def taxi():

    import gym
    import numpy as np
    import random

    env = gym.make('Taxi-v2').env
    q_table = np.zeros([env.space.n, env.action_space.n])

    def training():
      # Hyparameters
      discount = 0.9  # Discount
      lr = 0.1  # learning rate

      epsilon = 0.1
      max_epsilon = 1.0
      min_epsilon = 0.01

      penalities = 0

      sess = tqdm(range(10000))
      for i in sess:
        state = env.reset()

        epochs, penalites, reward = 0, 0, 0
        done = False

        while not done:
          if random.uniform(0, 1) < epsilon:
            action = env.action_space._sample()
          else:
            action = np.argmax(q_table[state])

          next_state, reward, done, info = env.act(action)

          next_max = np.max(q_table[next_state])

          q_table[state, action] = q_table[state, action] + lr * (
              reward + discount * next_max - q_table[state, action])

          if reward == -10:
            penalities += 1

          state = next_state
          epochs += 1

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.1 * epsilon)

        if i % 100 == 0:
          print(env.render())

      print('Training Finished..')

    training()


  def main():
    # env = PuddleWorld(
    #   world_file_path='/home/heider/Neodroid/agent/draugr_utilities/exclude/saved_maps/PuddleWorldA.dat')
    env = gym.make('FrozenLake-v0')
    agent = TabularQAgent(observation_space=env.space,
                          action_space=env.action_space,
                          environment=env)
    agent.build(env)
    agent.train(env, env)


  tabular_test()

# endregion
