#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = ''

import argparse

import draugr
from warg.arguments import add_bool_arg


def parse_arguments(agent_description, default_config: draugr.NOD) -> draugr.NOD:
  parser = argparse.ArgumentParser(description=agent_description)
  parser.add_argument("--ENVIRONMENT_NAME",
                      "-E",
                      type=str,
                      default=default_config.ENVIRONMENT_NAME,
                      metavar="ENVIRONMENT_NAME",
                      help="Name of the environment to run",
                      )
  parser.add_argument("--PRETRAINED_PATH",
                      "-T",
                      metavar="PATH",
                      type=str,
                      default="",
                      help="Path of pre-trained model"
                      )
  add_bool_arg(parser,
               "render",
               dest="RENDER_ENVIRONMENT",
               default=default_config.RENDER_ENVIRONMENT,
               help="Rendering of the environment",
               )

  add_bool_arg(parser,
               "inference",
               dest="INFERENCE",
               default=False,
               help="Should be an inference session",
               )

  parser.add_argument("--NUM_WORKERS",
                      "-N",
                      type=int,
                      default=6,
                      metavar="NUM_WORKERS",
                      help="Number of threads for agent (default: 4)",
                      )
  add_bool_arg(parser,
               "connect_to_running",
               dest="CONNECT_TO_RUNNING",
               default=default_config.CONNECT_TO_RUNNING,
               help="Connect to already running simulation or start an instance",
               )
  parser.add_argument("--SEED",
                      "-S",
                      type=int,
                      default=default_config.SEED,
                      metavar="SEED",
                      help=f"Random seed (default: {default_config.SEED})"
                      )
  parser.add_argument("--VERBOSE",
                      "-V",
                      action="store_true",
                      default=default_config.VERBOSE,
                      help="Enable verbose debug prints"
                      )
  add_bool_arg(parser,
               "skip",
               dest="SKIP_CONFIRMATION",
               default=True,
               help="Skip confirmation of config to be used",
               )
  parser.add_argument("--ROLLOUTS",
                      "-rollouts",
                      type=int,
                      default=default_config.ROLLOUTS,
                      metavar="ROLLOUTS",
                      help="Number of rollouts"
                      )
  parser.add_argument("--CONFIG",
                      "-config",
                      type=str,
                      default=None,
                      metavar="CONFIG",
                      help="Path to a config (nullifies all other arguments, if specified)",
                      )
  add_bool_arg(parser,
               "cuda",
               dest="USE_CUDA",
               default=default_config.USE_CUDA,
               help="Cuda flag")

  args = draugr.NOD(parser.parse_args().__dict__)

  return args
