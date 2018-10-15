#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path, PosixPath

__author__ = 'cnheider'


class ConfigObject(object):
  pass


def to_lower_properties(C_dict):
  if not type(C_dict) is dict:
    C_dict = to_dict(C_dict)

  a = ConfigObject()

  for (k, v) in C_dict.items():
    assert type(k) is str
    lowered = k.lower()
    if type(v) is PosixPath or type(v) is Path:
      setattr(a, lowered, str(v))
    else:
      setattr(a, lowered, v)

  return a


def get_upper_case_vars_or_protected_of(module):
  v = vars(module)
  if v:
    return {
      key:value
      for key, value in module.__dict__.items()
      if key.isupper() or (key.startswith('_') and not key.endswith('_'))
      }
  return {}

def to_dict(C,only_upper_case=True):
  if only_upper_case:
    return get_upper_case_vars_or_protected_of(C)
  else:
    return vars(C)


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
  parser.add_argument(
      '--CONFIG',
      '-config',
      type=str,
      default=None,
      metavar='CONFIG',
      help='Path to a config (nullifies all other arguments, if specified)')
  args = parser.parse_args()

  return args
