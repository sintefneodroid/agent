#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

import numpy as np

from utilities.torch_utilities.initialisation.ortho_weight_init import ortho_weights


def atari_initializer(module):
  ''' Parameter initializer for Atari models

Initializes Linear, Conv2d, and LSTM weights.
'''
  classname = module.__class__.__name__

  if classname == 'Linear':
    module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
    module.bias.data.zero_()

  elif classname == 'Conv2d':
    module.weight.data = ortho_weights(module.weight.data.size(), scale=np.sqrt(2.))
    module.bias.data.zero_()

  elif classname == 'LSTM':
    for name, param in module.named_parameters():
      if 'weight_ih' in name:
        param.data = ortho_weights(param.data.size(), scale=1.)
      if 'weight_hh' in name:
        param.data = ortho_weights(param.data.size(), scale=1.)
      if 'bias' in name:
        param.data.zero_()
