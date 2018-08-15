from collections import defaultdict
from itertools import count
from typing import Any, Tuple

import gym
import numpy as np
from neodroid import Displayable

from agents.abstract.agent import Agent


class TabularQAgent(Agent):
  '''
  Agent implementing tabular Q-learning.
  '''

  def update(self, *args, **kwargs) -> None:
    pass

  def _optimise_wrt(self, error, *args, **kwargs) -> None:
    pass

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def load(self, *args, **kwargs) -> None:
    pass

  def save(self, *args, **kwargs) -> None:
    pass

  def _build(self) -> None:
    pass

  def _defaults(self) -> None:

    self._action_n = 4

    self._eps = 0.8
    self._init_mean = 0.0
    self._init_std = 0.1
    self._learning_rate = .8
    self._discount_factor = .95

    self._q_table = defaultdict(
        lambda:self._init_std * np.random.randn(self._action_n) + self._init_mean)
    # Q = np.zeros([self._environment.observation_space.n, self._environment.action_space.n])

  def sample_action(self, observation, **kwargs):
    if type(observation) is not str:
      observation = str(observation)

    s = np.random.random()
    if s > self._eps:
      # action = self._environment._action_space.discrete_sample()
      action = self._environment.action_space.sample()
    else:
      action = np.argmax(self._q_table[observation])

    self._eps = self._eps * 0.9999

    return action

  def rollout(self, initial_state, environment, *, train=True, render=False, **kwargs) -> Any:
    obs = initial_state
    ep_r = 0
    steps = 0
    for t in count():
      action = self.sample_action(obs)
      next_obs, reward, done, _ = environment.step(action)
      next_obs = str(next_obs)

      current_q = self._q_table[obs][action]
      future = np.max(self._q_table[next_obs])
      exp_q = reward + self._discount_factor * future
      diff = self._learning_rate * (exp_q - current_q)
      self._q_table[obs][action] = current_q + diff
      #        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])

      obs = next_obs
      ep_r += reward

      if done:
        break

    return ep_r, steps

  def _train(self, env, iters=10000, *args, **kwargs) -> Tuple[Any, Any]:
    obs = env.reset()
    obs = str(obs)
    for i in range(iters):
      ep_r, steps = self.rollout(obs, env)
      obs = env.reset()
      obs = str(obs)
      print(ep_r)

    return 0, 0


def get_actor_configuration(environment, candidate):
  state_ob, _ = environment.configure(state=candidate)
  if environment:
    goal_pos_x = environment.description.configurable('ActorTransformX_').configurable_value
    # goal_pos_y = environment.description.configurable('ActorTransformY_').configurable_value
    goal_pos_z = environment.description.configurable('ActorTransformZ_').configurable_value
    return goal_pos_x, goal_pos_z


actor_configurations = []
success_estimates = []


def display_actor_configuration(env, candidate, frontier_displayer_name='FrontierPlotDisplayer'):
  actor_configuration = get_actor_configuration(env, candidate)
  vec3 = (actor_configuration[0], 0,
          actor_configuration[1])
  actor_configurations.append(vec3)
  est = 1
  success_estimates.append(est)
  frontier_displayable = [Displayable(frontier_displayer_name, (success_estimates, actor_configurations))]
  state_ob, info = env.display(frontier_displayable)


def main():
  # env = PuddleWorld(
  #   world_file_path='/home/heider/Neodroid/agent/utilities/exclude/saved_maps/PuddleWorldA.dat')
  env = gym.make('FrozenLake-v0')
  agent = TabularQAgent(observation_space=env.observation_space, action_space=env.action_space,
                        environment=env)
  agent.train(env)


if __name__ == '__main__':
  main()
