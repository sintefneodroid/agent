#!/usr/bin/env python3
# coding=utf-8
import torch

__author__ = 'cnheider'
import glob
import os


def main():
  '''

'''

  # _environment = neo.make(C.ENVIRONMENT_NAME, connect_to_running=C.CONNECT_TO_RUNNING)
  import configs.pg_config1 as C
  from agents.pg_agent import PGAgent
  from utilities.environment_wrappers.action_encoding import BinaryActionEncodingWrapper

  _environment = BinaryActionEncodingWrapper(
      name=C.ENVIRONMENT_NAME, connect_to_running=C.CONNECT_TO_RUNNING
      )
  _environment.seed(C.SEED)

  _list_of_files = glob.glob(str(C.MODEL_DIRECTORY) + '/*.model')
  _latest_model = max(_list_of_files, key=os.path.getctime)

  device = torch.device('cuda' if C.USE_CUDA else 'cpu')

  _agent = PGAgent(C)
  _agent.build_agent(_environment, device)
  _agent.load_model(_latest_model, evaluation=True)

  _agent.infer(_environment)


if __name__ == '__main__':
  main()
