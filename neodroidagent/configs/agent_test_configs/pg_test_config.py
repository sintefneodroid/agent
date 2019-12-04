#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

EnvironmentType = False
RENDER_ENVIRONMENT = True

EVALUATION_FUNCTION = torch.nn.CrossEntropyLoss

# Architecture
POLICY_ARCH_SPEC = GDKC(
    CategoricalMLP,
    NOD(
        **{
            "input_shape": None,  # Obtain from environment
            "hidden_layer_activation": torch.sigmoid,
            "hidden_layers": None,
            "output_shape": None,  # Obtain from environment
            "use_bias": True,
        }
    ),
)
