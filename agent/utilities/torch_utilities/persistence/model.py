#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pathlib

from warg.named_ordered_dictionary import NOD

__author__ = 'cnheider'
import datetime
import os
import shutil
import sys

import torch


def load_latest_model(model_directory,**kwargs):
  _list_of_files = model_directory.glob('*')
  _latest_model = max(_list_of_files, key=os.path.getctime)
  print('loading previous model: ' + _latest_model)

  return torch.load(_latest_model)


def ensure_directory_exist(model_path: pathlib.Path):
  if not model_path.exists():
    model_path.mkdir(parents=True)


def save_model(model, *, name='', **kwargs):
  kwargs = NOD(kwargs)

  model_date = datetime.datetime.now()
  prepend = ''
  if len(name) > 0:
    prepend = f'{name}-'
  model_name = (f'{prepend}{kwargs.project_name}-'
                f'{kwargs.config_name.replace(".", "_")}-'
                f'{model_date.strftime("%y%m%d%H%M")}.model')

  ensure_directory_exist(kwargs.model_directory)
  model_path = kwargs.model_directory / model_name

  ensure_directory_exist(kwargs.config_directory)
  config_path = pathlib.Path(kwargs.config_directory) / model_name
  try:
    save_model_and_configuration(model=model,
                                 model_path=model_path,
                                 config_path=config_path,
                                 **kwargs)
  except FileNotFoundError as e:
    print(e)
    saved = False
    while not saved:
      file_path = input('Enter another file path: ')
      model_path = pathlib.Path(file_path) / model_name
      try:
        saved = save_model_and_configuration(model=model,
                                             model_path=model_path,
                                             config_path=config_path,
                                             **kwargs)
      except FileNotFoundError as e:
        print(e)
        saved = False

  print(f'Saved model at {model_path}')


def save_model_and_configuration(*,
                                 model,
                                 model_path,
                                 config_path,
                                 **kwargs):
  if model and model_path:
    torch.save(model.state_dict(), model_path)
    save_config(config_path, **kwargs)
    return True
  return False


def save_config(new_path, config_file, **kwargs):
  config_path = pathlib.Path(config_file).absolute().parent / config_file

  shutil.copyfile(str(config_path), str(new_path) + '.py')


def convert_to_cpu(path=''):
  model = torch.load(path, map_location=lambda storage, loc:storage)
  torch.save(model, path + '.cpu')


if __name__ == '__main__':
  convert_to_cpu(sys.argv[1])
