#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib
from shutil import rmtree

from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'
__doc__ = ''


def main(keep_alive=True):
  from draugr.writers.tensorboards.launcher import launch_tensorboard
  from contextlib import suppress
  from time import sleep

  from agent.version import PROJECT_APP_PATH



  import argparse

  parser = argparse.ArgumentParser(description="Option for launching tensorboard")
  parser.add_argument('--clean', action='store_true', default=False,
                      help="Adding clean will wipe tensorboard logs")
  args = parser.parse_args()

  log_dir = str(PROJECT_APP_PATH.user_log)

  if args.clean:
    print(f'Wiping {PROJECT_APP_PATH.user_log}')
    if PROJECT_APP_PATH.user_log.exists():
      rmtree(log_dir)
    else:
      PROJECT_APP_PATH.user_log.mkdir()


  address = launch_tensorboard(log_dir)

  if keep_alive:
    print(f'tensorboard address: {address} for log_dir {log_dir}')
    with suppress(KeyboardInterrupt):
      while True:
        sleep(100)
  else:
    return address


if __name__ == '__main__':
  main()
