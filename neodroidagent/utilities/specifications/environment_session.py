#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import abc
import inspect
from typing import Union, Type

from neodroid.environments.environment import Environment
from neodroidagent.exceptions.exceptions import NoEnvironment
from neodroidagent.procedures.training.episodic import Episodic
from neodroidagent.utilities.specifications.procedure_specification import Procedure

__author__ = 'Christian Heider Nielsen'
__doc__ = ''


class EnvironmentSession(abc.ABC):

  def __init__(self,
               environment:Union[str, Environment],
               procedure: Union[Type[Procedure],Procedure] = Episodic,
               **kwargs):
    if isinstance(environment, str):
      raise NoEnvironment
    assert isinstance(environment, Environment)
    self._environment = environment
    self._procedure = procedure


  @abc.abstractmethod
  def __call__(self, agent, **kwargs):
    '''
    Start a session, builds Agent and starts/connect environment(s), and runs Procedure


    :param args:
    :param kwargs:
    :return:
    '''

    pass


if __name__ == '__main__':
  print(EnvironmentSession)
