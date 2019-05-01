#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'
import datetime
import os
import shutil
import sys

import torch


def load_latest_model(configuration):
  _list_of_files = configuration.MODEL_DIRECTORY.glob('*')
  _latest_model = max(_list_of_files, key=os.path.getctime)
  print('loading previous model: ' + _latest_model)

  return torch.load(_latest_model)


def save_model(model, configuration, *, name=''):
  model_date = datetime.datetime.now()
  prepend = ''
  if len(name) > 0:
    prepend = f'{name}-'
  model_name = (prepend +
                f'{configuration.PROJECT}-'
                f'{configuration.CONFIG_NAME.replace(".", "_")}-'
                f'{model_date.strftime("%y%m%d%H%M")}.model')

  model_path = os.path.join(configuration.MODEL_DIRECTORY, model_name)
  try:
    save_model_and_configuration(model=model, model_path=model_path, configuration=configuration)
  except FileNotFoundError as e:
    print(e)
    saved = False
    while not saved:
      file_path = input('Enter another file path: ')
      model_path = os.path.join(file_path, model_name)
      try:
        saved = save_model_and_configuration(model=model, model_path=model_path, configuration=configuration)
      except FileNotFoundError as e:
        print(e)
        saved = False

  print(f'Saved model at {model_path}')


def save_model_and_configuration(*, model, model_path, configuration):
  if model and model_path:
    torch.save(model.state_dict(), model_path)  # TODO possible to .cpu() copy would be great
    save_config(model_path, configuration)
    return True
  return False


def save_config(model_path, configuration):
  config_path = os.path.join(
      configuration.CONFIG_DIRECTORY, configuration.CONFIG_FILE
      )
  shutil.copyfile(config_path, model_path + '.py')


def convert_to_cpu(path=''):
  model = torch.load(path, map_location=lambda storage, loc:storage)
  torch.save(model, path + '.cpu')


if __name__ == '__main__':
  convert_to_cpu(sys.argv[1])
