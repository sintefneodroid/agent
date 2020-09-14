#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
"""
Description: Config for training
Author: Christian Heider Nielsen
"""

from neodroidagent.configs.base_config import *

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

UPDATE_DIFFICULTY_INTERVAL = 1000
