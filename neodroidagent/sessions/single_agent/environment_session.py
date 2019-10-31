#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import time
import types
from typing import Type, Union

from draugr.stopping.stopping_key import add_early_stopping_key_combination
from draugr.torch_utilities import torch_seed
from neodroid.environments.environment import Environment
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.exceptions.exceptions import NoAgent, NoEnvironment
from neodroidagent.procedures.training.on_policy_episodic import OnPolicyEpisodic
from neodroidagent.utilities.specifications.procedure_specification import Procedure
from warg.named_ordered_dictionary import NOD

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


class EnvironmentSession(abc.ABC):

  def __init__(self,
               *,
               environment: Environment,
               procedure: Union[Type[Procedure], Procedure] = OnPolicyEpisodic,
               **kwargs):
    self._environment = environment
    self._procedure = procedure

  def __call__(self,
               agent: Type[TorchAgent],
               *,
               environment_name,  # not used
               load_time,
               seed,
               save_model: bool = True,
               load_previous_environment_model_if_available: bool = False,
               **kwargs):
    '''
    Start a session, builds Agent and starts/connect environment(s), and runs Procedure


    :param args:
    :param kwargs:
    :return:
    '''

    if agent is None:
      raise NoAgent

    if isinstance(agent, (types.ClassType)):
      torch_seed(seed)
      self._environment.seed(seed)

      agent = agent(environment_name=self._environment.environment_name,
                    load_time=load_time,
                    seed=seed,
                    **kwargs)
      agent.build(self._environment.observation_space,
                  self._environment.action_space,
                  self._environment.signal_space)

    agent_class_name = agent.__class__.__name__
    agent_models = '-'.join(agent.models.keys())

    total_shape = '_'.join([str(i) for i in (
      self._environment.observation_space.shape +
      self._environment.action_space.shape +
      self._environment.signal_space.shape)])

    base_model_dir = (PROJECT_APP_PATH.user_data /
                      self._environment.environment_name /
                      agent_class_name /
                      f'{agent_models}{total_shape}'
                      )
    base_config_dir = (PROJECT_APP_PATH.user_log /
                       self._environment.environment_name /
                       agent_class_name /
                       f'{agent_models}{total_shape}'
                       )

    model_directory = (base_model_dir / load_time)
    log_directory = (base_config_dir / load_time)

    if load_previous_environment_model_if_available:
      agent.load(model_path=base_model_dir, evaluation=False)

    listener = add_early_stopping_key_combination(self._procedure.stop_procedure,
                                                  **kwargs)

    proc = self._procedure(agent, environment=self._environment)

    training_start_timestamp = time.time()
    if listener:
      listener.start()

    try:
      training_resume = proc(
        log_directory=log_directory,
        environment_name=self._environment.environment_name,
        load_time=load_time,
        seed=seed,
        **kwargs)
      if training_resume and 'stats' in training_resume:
        training_resume.stats.save(
          log_directory=log_directory, **kwargs)

    except KeyboardInterrupt:
      pass
    time_elapsed = time.time() - training_start_timestamp

    if listener:
      listener.stop()

    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    line_width = 9
    print(f'\n{"-" * line_width} {end_message} {"-" * line_width}\n')

    if save_model:
      agent.save(save_directory=model_directory,
                 **kwargs)

    try:
      self._environment.close()
    except BrokenPipeError:
      pass

    exit(0)


if __name__ == '__main__':
  print(EnvironmentSession)
