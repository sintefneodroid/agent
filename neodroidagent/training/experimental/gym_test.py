#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = ''
import glob
import os

import gym

from neodroidagent.interfaces.specifications import TrainingSession


class agent_test_gym(TrainingSession):
  def __call__(self, *args, **kwargs):
    '''
      Executes training session
    '''

    import neodroidagent.configs.agent_test_configs.pg_test_config as C
    from neodroidagent.agents.model_free.policy_optimisation.pg_agent import PGAgent

    _environment = gym.make(C.ENVIRONMENT_NAME)
    _environment.seed(C.SEED)

    _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
    _latest_model = max(_list_of_files, key=os.path.getctime)

    _agent = PGAgent(C)
    _agent.build(_environment)
    _agent.load(_latest_model, evaluation=True)

    _agent.infer(_environment)
