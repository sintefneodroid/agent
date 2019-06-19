#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__ = 'cnheider'
__doc__ = ''


def main(keep_alive = True):

  from draugr.writers.tensorboards.launcher import launch_tensorboard
  from contextlib import suppress
  from time import sleep

  from agent.version import PROJECT_APP_PATH
  log_dir = str(PROJECT_APP_PATH.user_log)
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
