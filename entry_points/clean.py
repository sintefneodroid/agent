#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from shutil import rmtree

from agent import PROJECT_APP_PATH

__author__ = 'cnheider'
__doc__ = r'''This script clean all Neodroid Agent related data from the environment'''


def main():
  print(f'Wiping {PROJECT_APP_PATH.user_log}')
  if PROJECT_APP_PATH.user_log.exists():
    log_dir = str(PROJECT_APP_PATH.user_log)
    rmtree(log_dir)
  else:
    PROJECT_APP_PATH.user_log.mkdir()

  print(f'Wiping {PROJECT_APP_PATH.user_data}')
  if PROJECT_APP_PATH.user_data.exists():
    data_dir = str(PROJECT_APP_PATH.user_data)
    rmtree(data_dir)
  else:
    PROJECT_APP_PATH.user_data.mkdir()

  print(f'Wiping {PROJECT_APP_PATH.user_cache}')
  if PROJECT_APP_PATH.user_cache.exists():
    cache_dir = str(PROJECT_APP_PATH.user_cache)
    rmtree(cache_dir)
  else:
    PROJECT_APP_PATH.user_cache.mkdir()

  print(f'Wiping {PROJECT_APP_PATH.user_config}')
  if PROJECT_APP_PATH.user_config.exists():
    config_dir = str(PROJECT_APP_PATH.user_config)
    rmtree(config_dir)
  else:
    PROJECT_APP_PATH.user_config.mkdir()


if __name__ == '__main__':
  main()
