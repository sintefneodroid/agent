__author__ = "Christian Heider Nielsen"
__doc__ = """
Description: Config for training
Author: Christian Heider Nielsen
"""

from neodroidagent.configs.base_config import *

CONFIG_NAME = f"{__name__} on {CONFIG_NAME}"

# noinspection PyUnresolvedReferences
from neodroidagent.configs.base_config import *

# ENVIRONMENT_NAME = 'LunarLanderContinuous-v2'

ENVIRONMENT_NAME = "Pendulum-v1"
# ENVIRONMENT_NAME = "BipedalWalker-v2"
# ENVIRONMENT_NAME = "LunarLanderContinuous-v2"
# ENVIRONMENT_NAME = 'MountainCarContinuous-v0'
