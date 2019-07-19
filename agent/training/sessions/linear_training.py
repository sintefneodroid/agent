#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from collections import Iterable
from itertools import count
from typing import Type

import gym

import draugr
from agent import utilities as U
from agent.interfaces.agent import Agent
from agent.interfaces.specifications import TrainingSession
from draugr.stopping_key import add_early_stopping_key_combination
from neodroid.environments.wrappers import NeodroidGymWrapper
from neodroid.environments.wrappers.vector_environment import VectorEnvironment
from trolls.wrappers.vector_environments import VectorWrap

__author__ = 'cnheider'
__doc__ = ''


class linear_training(TrainingSession):

  def __call__(self,
               agent_type: Type[Agent],
               *,
               environment=None,
               save=False,
               has_x_server=False,
               **kwargs):

    kwargs = draugr.NOD(**kwargs)

    if not kwargs.connect_to_running:
      if not environment:
        if '-v' in kwargs.environment_name:
          environment = VectorWrap(NeodroidGymWrapper(gym.make(kwargs.environment_name)))
        else:
          environment = VectorEnvironment(name=kwargs.environment_name,
                                          connect_to_running=kwargs.connect_to_running)
    else:
      environment = VectorEnvironment(name=kwargs.environment_name,
                                      connect_to_running=kwargs.connect_to_running)

    U.set_seeds(kwargs['SEED'])
    environment.seed(kwargs['SEED'])

    agent = agent_type(**kwargs)
    agent.build(environment)

    listener = add_early_stopping_key_combination(agent.stop_training, has_x_server=save)

    if listener:
      listener.start()
    try:
      training_resume = self._training_procedure(agent,
                                                 environment,
                                                 render=kwargs.render_environment,
                                                 **kwargs)
    finally:
      if listener:
        listener.stop()

    if save:
      identifier = count()
      if isinstance(training_resume.models, Iterable):
        for model in training_resume.models:
          U.save_model(model, name=f'{agent.__class__.__name__}-{identifier.__next__()}', **kwargs)
      else:
        U.save_model(training_resume.models,
                     name=f'{agent.__class__.__name__}-{identifier.__next__()}', **kwargs)

      if training_resume.stats:
        training_resume.stats.save(project_name=kwargs.project,
                                   config_name=kwargs.config_name,
                                   directory=kwargs.log_directory)

    environment.close()
