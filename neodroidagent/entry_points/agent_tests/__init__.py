#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 20/01/2020
           """

from neodroidagent.entry_points.agent_tests.torch_agent_tests.ddpg_test import *
from neodroidagent.entry_points.agent_tests.torch_agent_tests.dqn_test import *
from neodroidagent.entry_points.agent_tests.torch_agent_tests.pg_test import *
from neodroidagent.entry_points.agent_tests.torch_agent_tests.ppo_test import *
from .random_test import *
from neodroidagent.entry_points.agent_tests.torch_agent_tests.sac_test import *

from .torch_agent_tests.ddpg_test import ddpg_config, ddpg_run
from .torch_agent_tests.dqn_test import dqn_config, dqn_run
from .torch_agent_tests.pg_test import pg_config, pg_run
from .torch_agent_tests.ppo_test import ppo_config, ppo_run
from .torch_agent_tests.sac_test import sac_config, sac_run

AGENT_OPTIONS = {
    "ddpg-gym": ddpg_gym_test,
    "ddpg": ddpg_run,
    "ppo-gym": ppo_gym_test,
    "ppo": ppo_run,
    "pg-gym": pg_gym_test,
    "pg": pg_run,
    "dqn-gym": dqn_gym_test,
    "dqn": dqn_run,
    "sac-gym": sac_gym_test,
    "sac": sac_run,
    "random-gym": random_gym_test,
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
