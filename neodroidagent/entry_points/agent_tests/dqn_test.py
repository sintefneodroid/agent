#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Union

from neodroidagent.agents import DQNAgent
from neodroidagent.common import (
    OffPolicyEpisodic,
    ParallelSession,
    TransitionPointBuffer,
)
from neodroidagent.common.architectures.mlp_variants import SingleHeadMLP
from neodroidagent.configs.test_reference.base_dicrete_test_config import *

__author__ = "Christian Heider Nielsen"

from neodroidagent.entry_points.session_factory import session_factory

from neodroidagent.utilities.exploration.exploration_specification import (
    ExplorationSpecification,
)

"""
Description: Config for training
Author: Christian Heider Nielsen
"""

CONFIG_NAME = __name__
import pathlib

CONFIG_FILE_PATH = pathlib.Path(__file__)

MEMORY = TransitionPointBuffer(int(1e5))
DOUBLE_DQN = True
LEARNING_FREQUENCY = 4
BATCH_SIZE = 128
EXPLORATION_DECAY = ITERATIONS * 5
EXPLORATION_SPEC = ExplorationSpecification(0.95, 0.05, EXPLORATION_DECAY)

DISCOUNT_FACTOR = 0.999
RENDER_ENVIRONMENT = True

TRAIN_AGENT = True
ITERATIONS = int(1e5)
SYNC_TARGET_MODEL_FREQUENCY = 0
INITIAL_OBSERVATION_PERIOD = 100
UPDATE_TARGET_PERCENTAGE = 1 / 10

CONTINUE = True

OPTIMISER_SPEC = GDKC(torch.optim.Adam, lr=1e-2)
SCHEDULER_SPEC = None

RENDER_FREQUENCY = 1

# Architecture
VALUE_ARCH_SPEC = GDKC(
    SingleHeadMLP,
    hidden_layers=None,
    hidden_layer_activation=torch.tanh,
    use_dropout=False,
)

dqn_config = globals()


def dqn_run(
    skip_confirmation: bool = True,
    environment_type: Union[bool, str] = True,
    config=dqn_config,
):
    session_factory(
        DQNAgent,
        config,
        session=ParallelSession(
            environment_name=ENVIRONMENT_NAME,
            procedure=OffPolicyEpisodic,
            # auto_reset_on_terminal_state=True,
            environment=environment_type,
        ),
        skip_confirmation=skip_confirmation,
        environment=environment_type,
    )


def dqn_test(config=dqn_config):
    dqn_run(environment_type="gym", config=config)


if __name__ == "__main__":
    dqn_test()
