#!/usr/bin/env python3
# coding=utf-8
import argparse

__author__ = 'cnheider'


def parse_arguments(desc, C):
  parser = argparse.ArgumentParser(description=desc)
  parser.add_argument(
      '--ENVIRONMENT_NAME',
      '-E',
      type=str,
      default=C.ENVIRONMENT_NAME,
      metavar='ENVIRONMENT_NAME',
      help='Name of the environment to run')
  parser.add_argument(
      '--PRETRAINED_PATH',
      '-T',
      metavar='PATH',
      type=str,
      default='',
      help='Path of pre-trained model')
  parser.add_argument(
      '--RENDER_ENVIRONMENT',
      '-R',
      action='store_true',
      default=C.RENDER_ENVIRONMENT,
      help='Render the environment')
  parser.add_argument(
      '--NUM_WORKERS',
      '-N',
      type=int,
      default=4,
      metavar='NUM_WORKERS',
      help='Number of threads for agent (default: 4)')
  parser.add_argument(
      '--CONNECT_TO_RUNNING',
      '-C',
      action='store_true',
      default=C.CONNECT_TO_RUNNING,
      help='Connect to already running environment instead of starting another instance')
  parser.add_argument(
      '--SEED',
      '-S',
      type=int,
      default=C.SEED,
      metavar='SEED',
      help=f'Random seed (default: {C.SEED})')
  parser.add_argument(
      '--VERBOSE',
      '-V',
      action='store_true',
      default=C.VERBOSE,
      help='Enable verbose debug prints')
  parser.add_argument(
      '--skip_confirmation',
      '-skip',
      action='store_true',
      default=False,
      help='Skip confirmation of config to be used')
  parser.add_argument(
      '--ROLLOUTS',
      '-rollouts',
      type=int,
      default=C.ROLLOUTS,
      metavar='ROLLOUTS',
      help='Number of rollouts')
  args = parser.parse_args()

  return args
