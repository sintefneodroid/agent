#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import Type

import torch

import draugr
from agent.exceptions.exceptions import NoTrainingProcedure
from agent.interfaces.agent import Agent
from agent.interfaces.specifications import TrainingSession
from agent.training.arguments import parse_arguments
from agent.training.procedures import train_episodically
from agent.training.sessions import linear_training
from neodroid.environments.wrappers.vector_environment import VectorEnvironment
from warg.arguments import config_to_mapping

__author__ = 'cnheider'
__doc__ = ''


def agent_session_entry_point(agent: Type[Agent],
                              config: object,
                              *,
                              training_session: TrainingSession = linear_training,
                              parse_args: bool = True,
                              save: bool = True,
                              has_x_server: bool = True,
                              skip_confirmation: bool = True
                              ):
  r'''
    Entry point start a starting a training session with the functionality of parsing cmdline arguments and confirming configuration to use before training and overwriting of default training configurations
  '''

  if training_session is None:
    raise NoTrainingProcedure
  elif isinstance(training_session, type):
    training_session = training_session(training_procedure=train_episodically)

  if parse_args:
    args = parse_arguments(f'{type(agent)}', draugr.NOD(config.__dict__))

    skip_confirmation = args.SKIP_CONFIRMATION

    if args.PRETRAINED_PATH != '':
      pass

    if args.INFERENCE:
      pass

    # TODO: load earlier model and inference flags

    if 'CONFIG' in args.keys() and args['CONFIG']:
      import importlib.util
      spec = importlib.util.spec_from_file_location('overloaded.config', args['CONFIG'])
      config = importlib.util.module_from_spec(spec)
      spec.loader.exec_module(config)
    else:
      for key, arg in args.items():
        if key != 'CONFIG':
          setattr(config, key, arg)

  if has_x_server:
    display_env = os.getenv('DISPLAY', None)
    if display_env is None:
      config.RENDER_ENVIRONMENT = False
      has_x_server = False

  config_mapping = config_to_mapping(config)

  if not skip_confirmation:
    draugr.sprint(f'\nUsing config: {config}\n', highlight=True, color='yellow')
    for key, arg in config_mapping:
      print(f'{key} = {arg}')

    draugr.sprint(f'\n.. Also save:{save}, has_x_server:{has_x_server}')
    input('\nPress Enter to begin... ')

  try:
    training_session(agent,
                     save=save,
                     has_x_server=has_x_server,
                     **config_mapping)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()

  exit(0)




