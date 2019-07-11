#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroid.models import Configuration, Displayable, Reaction, ReactionParameters

__author__ = 'cnheider'
import random
from types import coroutine

import numpy as np
from tqdm import tqdm


@coroutine
def grid_world_sample_entire_configuration_space(environment):
  if environment:
    actor_x_conf = environment.description.configurable('Vertical')
    # actor_y_conf = environment.description.configurable('Orthogonal')
    actor_z_conf = environment.description.configurable('Horizontal')
    x_space = actor_x_conf.configurable_space
    # y_space = actor_y_conf.configurable_space
    z_space = actor_z_conf.configurable_space
    x_steps = x_space.discrete_steps
    z_steps = z_space.discrete_steps
    x_min, x_max = x_space.min_value, x_space.max_value
    z_min, z_max = z_space.min_value, z_space.max_value
    for x in np.linspace(x_min, x_max, x_steps):
      for z in np.linspace(z_min, z_max, z_steps):
        # for y in np.linspace(y_space.min_value, y_space.max_value, y_space.discrete_steps):
        initial_configuration = [
          Configuration('Vertical', x),
          # Configuration('Orthogonal', y),
          Configuration('Horizontal', z),
          ]

        yield initial_configuration
  return


@coroutine
def grid_world_random_sample_uniformly_entire_configuration_space(environment):
  if environment:
    initial_configurations = []
    actor_x_conf = environment.description.configurable('Vertical')
    # actor_y_conf = environment.description.configurable('Orthogonal')
    actor_z_conf = environment.description.configurable('Horizontal')
    x_space = actor_x_conf.configurable_space
    # y_space = actor_y_conf.configurable_space
    z_space = actor_z_conf.configurable_space
    for x in np.linspace(x_space.min_value, x_space.max_value, x_space.discrete_steps):
      for z in np.linspace(z_space.min_value, z_space.max_value, z_space.discrete_steps):
        # for y in np.linspace(y_space.min_value, y_space.max_value, y_space.discrete_steps):
        initial_configuration = [
          Configuration('Vertical', x),
          # Configuration('Orthogonal', y),
          Configuration('Horizontal', z),
          ]
        initial_configurations.append(initial_configuration)

    while 1:
      yield random.sample(initial_configurations)
  return


def estimate_entire_state_space(env,
                                agent,
                                C,
                                *,
                                save_snapshot,
                                statistics=None,
                                displayer_name='ScatterPlot'):
  actor_configurations = []
  success_estimates = []
  displayables = []
  for configuration in grid_world_sample_entire_configuration_space(env):
    configure_params = ReactionParameters(terminable=True,
                                          episode_count=False,
                                          reset=True,
                                          configure=True,
                                          )

    conf_reaction = Reaction(parameters=configure_params,
                             configurations=configuration,
                             displayables=displayables
                             )

    displayables = [Displayable(displayer_name, (success_estimates, actor_configurations))]

    env.reset()
    state_ob, info = env.configure(conf_reaction)
    if not info.terminated:
      est, _, _ = estimate_initial_state_expected_return(info,
                                                         env,
                                                         agent,
                                                         C,
                                                         save_snapshot=save_snapshot,
                                                         statistics=statistics,
                                                         train=False,
                                                         random_sample=False)

      vec3 = (configuration[0].configurable_value,
              0,  # configuration[1].configurable_value,
              configuration[1].configurable_value  # configuration[2].configurable_value
              )
      actor_configurations.append(vec3)
      success_estimates.append(est)

  displayables = [Displayable(displayer_name, (success_estimates, actor_configurations))]
  conf_reaction = Reaction(displayables=displayables)
  _ = env.configure(conf_reaction)


_episode_i = 0
_step_i = 0


def estimate_initial_state_expected_return(candidate,
                                           env,
                                           agent,
                                           C,
                                           *,
                                           save_snapshot=False,
                                           statistics=None,
                                           random_sample=False,
                                           train=False):
  global _step_i, _episode_i

  N_c_r = C.CANDIDATE_ROLLOUTS

  rollout_signals = 0
  rollout_session = range(1, N_c_r + 1)
  rollout_session = tqdm(rollout_session, leave=False, disable=False)
  for j in rollout_session:
    rollout_session.set_description(f'Candidate rollout #{j} of {N_c_r} | '
                                    f'Est: {rollout_signals / N_c_r}'
                                    )
    state_ob, _ = env.configure(state=candidate)

    signals, steps, *stats = agent.rollout(state_ob, env, train=train, disallow_random_sample=random_sample)
    rollout_signals += signals

    if train:
      _step_i += steps
      _episode_i += 1

    if statistics:
      statistics.signals.append(signals)
      statistics.lengths.append(steps)
      statistics.entropies.append(stats[0])

    if _episode_i % C.SAVE_MODEL_INTERVAL == 0:
      pass
      # save_snapshot()

  return rollout_signals / N_c_r, _episode_i, _step_i


def display_actor_configurations(env, candidates, frontier_displayer_name='ScatterPlot2'):
  actor_configurations = []
  for candidate in candidates:
    actor_configuration = get_actor_configuration(env, candidate)
    vec3 = (actor_configuration[0],
            0,
            actor_configuration[1])
    actor_configurations.append(vec3)
  frontier_displayable = [Displayable(frontier_displayer_name,
                                      ([1] * len(actor_configuration),
                                       actor_configurations))]
  state_ob, info = env.display(frontier_displayable)


def get_initial_configuration_from_goal(environment):
  state = environment.describe()
  if environment:
    goal_pos_x = environment.description.configurable('GoalX').configurable_value
    # goal_pos_y = environment.description.configurable('GoalY').configurable_value
    goal_pos_z = environment.description.configurable('GoalZ').configurable_value
    initial_configuration = (Configuration('Vertical', goal_pos_x),
                             # Configuration('Orthogonal', goal_pos_y),
                             Configuration('Horizontal', goal_pos_z),
                             )
    return initial_configuration


def get_actor_configuration(environment, candidate):
  state_ob, _ = environment.configure(state=candidate)
  # state = environment.describe()
  if environment:
    goal_pos_x = environment.description.configurable('Vertical').configurable_value
    # goal_pos_y = environment.description.configurable('GoalY').configurable_value
    goal_pos_z = environment.description.configurable('Horizontal').configurable_value
    return goal_pos_x, goal_pos_z
