#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base_test_config import *

__author__ = 'cnheider'

CONFIG_NAME = __name__
CONFIG_FILE = __file__

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True

EVALUATION_FUNCTION = torch.nn.CrossEntropyLoss

OPTIMISER_SPEC = GDCS(torch.optim.Adam,
                      NOD(lr=3e-5,
                          weight_decay=3e-9,
                          eps=3e-4))

DISCOUNT_FACTOR = 0.95
PG_ENTROPY_REG = 3e-9

# Architecture
POLICY_ARCH_SPEC = GDCS(CategoricalMLP,
                        NOD(**{
                          'input_shape':            None,  # Obtain from environment
                          'hidden_layer_activation':torch.relu,
                          'hidden_layers':          None,
                          'output_shape':           None,  # Obtain from environment
                          'use_bias':               True,
                          })
                        )
