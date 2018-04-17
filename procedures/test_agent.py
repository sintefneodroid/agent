#!/usr/bin/env python3
# coding=utf-8
__author__ = 'cnheider'
import glob
import os

import configs.curriculum.curriculum_config as C
from agents.pg_agent import PGAgent
from utilities.environment_wrappers.action_encoding import BinaryActionEncodingWrapper


def main():
  """

  """
  # _environment = neo.make(C.ENVIRONMENT, connect_to_running=C.CONNECT_TO_RUNNING)
  _environment = BinaryActionEncodingWrapper(name=C.ENVIRONMENT, connect_to_running=C.CONNECT_TO_RUNNING)
  _environment.seed(C.RANDOM_SEED)

  # C.ARCH_PARAMS['input_size'] = [4]
  # C.ARCH_PARAMS['output_size'] = [_environment.action_space.n]

  _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
  _latest_model = max(_list_of_files, key=os.path.getctime)

  # _agent = DDPGAgent(C)
  # _agent = DQNAgent(C)
  _agent = PGAgent(C)
  _agent.build_model(_environment)
  _agent.load_model(C, _latest_model)

  _agent.infer(_environment)


if __name__ == '__main__':
  main()
