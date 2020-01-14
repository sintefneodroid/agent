#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from neodroidagent.common.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.common.configs.base_dicrete_test_config import *
import pathlib

__author__ = "Christian Heider Nielsen"

CONFIG_NAME = __name__

CONFIG_FILE_PATH = pathlib.Path(__file__)

EnvironmentType = False
RENDER_ENVIRONMENT = True

# Architecture
POLICY_ARCH_SPEC = GDKC(
    CategoricalMLP,
    hidden_layers=(128,),
    hidden_layer_activation=torch.relu,
    use_bias=True,
)
