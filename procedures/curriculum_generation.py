#!/usr/bin/env python3
# coding=utf-8
__author__='cnheider'
from collections import namedtuple
from itertools import count

import numpy as np

import neodroid.wrappers.curriculum_wrapper as neo
from neodroid.models import Configuration

random_motion_horizon = 5
initial_states_to_generate = 100


def get_initial_configuration2(environment):
  if environment:
    # goal_pos_x = environment.description.configurable('GoalTransformX').observation
    # goal_pos_z = environment.description.configurable('GoalTransformZ').observation
    # return goal_pos_x,goal_pos_z
    goal_pos = environment.description.configurable('GoalPosition').observation
    initial_configuration = [Configuration('ActorPositionX', goal_pos[0]),
                             Configuration('ActorPositionY', goal_pos[1]),
                             Configuration('ActorPositionZ', goal_pos[2])]
    return initial_configuration


def get_initial_configuration(environment):
  if environment:
    goal_pos_x = environment.description.configurable('GoalTransformX').observation
    goal_pos_z = environment.description.configurable('GoalTransformZ').observation
    initial_configuration = [Configuration('ActorTransformX', goal_pos_x),
                             Configuration('ActorTransformZ', goal_pos_z)]
    return initial_configuration


def main():
  _environment = neo.make('grid_world', connect_to_running=False)
  _environment.seed(42)

  initial_configuration = get_initial_configuration(_environment)

  initial_start_set = _environment.generate_initial_states_from_configuration(initial_configuration)

  StateValuePair = namedtuple('StateValuePair', ('init_state', 'value_estimate'))
  frontier = []

  rollouts_for_each_state = 100
  low = 0.1
  high = 0.9

  for i in count(1):
    if not _environment.is_connected:
      break

    if i % 3 == 0:
      fs = sample_frontier(frontier)
      good_starts = [state for state, value in fs if low < value < high]
      if len(good_starts):
        initial_start_set = []
        for start in good_starts:
          initial_start_set.extend(_environment.generate_initial_states_from_state(start))

    init_state = sample_initial_state(initial_start_set)

    episode_rewards = []
    for j in range(rollouts_for_each_state):
      _environment.configure(state=init_state)
      episode_reward = 0
      for k in count(1):

        actions = _environment.action_space.sample()
        observations, reward, terminated, info = _environment.act(actions)

        episode_reward += reward
        if terminated:
          print('Interrupted', reward)
          episode_rewards.append(episode_reward)
          break

    frontier.append(StateValuePair(init_state, np.mean(episode_rewards)))

  _environment.close()


def sample_frontier(memory, size=5):
  new_frontier = []
  while len(new_frontier) < size:
    idx = np.random.randint(0, len(memory))
    new_frontier.append(memory[idx])
  return new_frontier


def sample_initial_state(memory):
  idx = np.random.randint(0, len(memory))
  return memory[idx]


if __name__ == '__main__':
  main()
