#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'

from configs.agent_test_configs.base_test_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

CONNECT_TO_RUNNING = False
RENDER_ENVIRONMENT = False

ROLLOUTS = 2000

EVALUATION_FUNCTION = torch.nn.CrossEntropyLoss

DISCOUNT_FACTOR = 0.99
OPTIMISER_LEARNING_RATE = 1e-4
PG_ENTROPY_REG = 1e-4

# Architecture
POLICY_ARCH_PARAMS = NOD(
    input_size=None,  # Obtain from environment
    hidden_layer_activation=F.leaky_relu,
    hidden_layers=[128, 64, 32, 16],
    output_size=None,  # Obtain from environment
    use_bias=True
    )
POLICY_ARCH = U.CategoricalCNN