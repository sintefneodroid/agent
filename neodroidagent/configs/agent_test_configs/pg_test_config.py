#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base_dicrete_test_config import *

__author__ = 'Christian Heider Nielsen'

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = True

EVALUATION_FUNCTION = torch.nn.CrossEntropyLoss

OPTIMISER_SPEC = GDKC(torch.optim.Adam,
                      NOD(lr=3e-4,
                          weight_decay=3e-11,
                          eps=3e-6))

DISCOUNT_FACTOR = 0.95
PG_ENTROPY_REG = 3e-9

# Architecture
POLICY_ARCH_SPEC = GDKC(CategoricalMLP,
                        NOD(**{
                          'input_shape':            None,  # Obtain from environment
                          'hidden_layer_activation':torch.relu,
                          'hidden_layers':          None,
                          'output_shape':           None,  # Obtain from environment
                          'use_bias':               True,
                          })
                        )
