#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import time
import types
from typing import Type, Union

from draugr.stopping.stopping_key import add_early_stopping_key_combination
from draugr.torch_utilities import set_seeds
from neodroid.environments.environment import Environment
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.exceptions.exceptions import NoAgent, NoEnvironment
from neodroidagent.procedures.training.episodic import Episodic
from neodroidagent.utilities.specifications.procedure_specification import Procedure
from warg.named_ordered_dictionary import NOD

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


class EnvironmentSession(abc.ABC):

  def __init__(self,
               environment: Environment,
               procedure: Union[Type[Procedure], Procedure] = Episodic,
               **kwargs):
    self._environment = environment
    self._procedure = procedure

  def __call__(self,
               agent: Type[TorchAgent],
               *,
               environment_name,
               load_time,
               seed,
               save_model: bool = True,
               **kwargs):
    '''
    Start a session, builds Agent and starts/connect environment(s), and runs Procedure


    :param args:
    :param kwargs:
    :return:
    '''

    if agent is None:
      raise NoAgent

    agent_class_name = agent.__name__
    model_directory = (PROJECT_APP_PATH.user_data / environment_name /
                       agent_class_name / load_time / 'models')
    log_directory = (PROJECT_APP_PATH.user_log / environment_name /
                     agent_class_name / load_time)

    if isinstance(agent, (types.ClassType)):
      set_seeds(seed)
      self._environment.seed(seed)

      agent = agent(environment_name=environment_name,
                    load_time=load_time,
                    seed=seed,
                    **kwargs)
      agent.build(self._environment.observation_space,
                  self._environment.action_space,
                  self._environment.signal_space)

    listener = add_early_stopping_key_combination(self._procedure.stop_procedure,**kwargs)

    proc = self._procedure(agent, self._environment)

    training_start_timestamp = time.time()
    if listener:
      listener.start()

    try:
      training_resume = proc(
                             log_directory=log_directory,
                             environment_name=environment_name,
                             load_time=load_time,
                             seed=seed,
                             **kwargs)
      if training_resume and 'stats' in training_resume:
        training_resume.stats.save(
                                   log_directory=log_directory,**kwargs)

    except KeyboardInterrupt:
      pass
    time_elapsed = time.time() - training_start_timestamp

    if listener:
      listener.stop()

    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    line_width = 9
    print(f'\n{"-" * line_width} {end_message} {"-" * line_width}\n')

    if save_model:
      agent.save(model_directory=model_directory,config_directory=model_directory, **kwargs)

    try:
      self._environment.close()
    except BrokenPipeError:
      pass

    exit(0)


if __name__ == '__main__':
  print(EnvironmentSession)
