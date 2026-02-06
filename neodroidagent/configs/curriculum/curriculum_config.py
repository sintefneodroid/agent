__author__ = "Christian Heider Nielsen"
__doc__ = """
Description: Config for training
Author: Christian Heider Nielsen
"""

from neodroidagent.configs.base_config import *

CONFIG_NAME = f"{__name__} on {CONFIG_NAME}"
from pathlib import Path

CONFIG_FILE_PATH = Path(__file__)

UPDATE_DIFFICULTY_INTERVAL = 1000
