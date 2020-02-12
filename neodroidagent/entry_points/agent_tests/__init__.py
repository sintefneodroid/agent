#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/01/2020
           """

from .ddpg_test import *
from .dqn_test import *
from .pg_test import *
from .ppo_test import *
from .random_test import *
from .sac_test import *

AGENT_OPTIONS = {
    "ddpg-gym": ddpg_test,
    "ddpg": ddpg_run,
    "ppo-gym": ppo_test,
    "ppo": ppo_run,
    "pg-gym": pg_test,
    "pg": pg_run,
    "dqn-gym": dqn_test,
    "dqn": dqn_run,
    "sac-gym": sac_test,
    "sac": sac_run,
    "random-gym": random_test,
    "random": random_run,
}

AGENT_CONFIG = {
    "ddpg-gym": ddpg_config,
    "ddpg": ddpg_config,
    "ppo-gym": ppo_config,
    "ppo": ppo_config,
    "pg-gym": pg_config,
    "pg": pg_config,
    "dqn-gym": dqn_config,
    "dqn": dqn_config,
    "sac-gym": sac_config,
    "sac": sac_config,
    "random-gym": random_config,
    "random": random_config,
}
