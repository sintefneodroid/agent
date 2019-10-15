#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Christian Heider Nielsen'
__doc__ = ''
import glob
import os

import gym

from neodroidagent.utilities.specifications import EnvironmentSession


class agent_test_gym(EnvironmentSession):
  def __call__(self, *args, **kwargs):
    '''
      Executes training session
    '''

    import neodroidagent.configs.agent_test_configs.pg_test_config as C
    from neodroidagent.agents.torch_agents.model_free import PGAgent

    _environment = gym.make(C.ENVIRONMENT_NAME)
    _environment.seed(C.SEED)

    _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
    _latest_model = max(_list_of_files, key=os.path.getctime)

    _agent = PGAgent(C)
    _agent.build(_environment)
    _agent.load(_latest_model, evaluation=True)

    _agent.infer(_environment)
