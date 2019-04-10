#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from neodroid.models import Displayable

__author__ = 'cnheider'


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
