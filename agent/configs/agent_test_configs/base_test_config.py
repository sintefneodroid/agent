#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = 'cnheider'
__doc__ = '''
Description: Config for training
Author: Christian Heider Nielsen
'''

from agent.configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

# Mujoco
# ENVIRONMENT_NAME = 'FetchPickAndPlace-v0'
# ENVIRONMENT_NAME = 'FetchReach-v1'
# ENVIRONMENT_NAME = 'FetchPush-v0'
# ENVIRONMENT_NAME = 'FetchSlide-v0'c
# ENVIRONMENT_NAME = 'MountainCarContinuous-v0'
# ENVIRONMENT_NAME = 'InvertedPendulum-v2'
# ENVIRONMENT_NAME = 'Reacher-v1'
# ENVIRONMENT_NAME = 'Hopper-v1'
# ENVIRONMENT_NAME = 'Ant-v1'
# ENVIRONMENT_NAME = 'Humanoid-v1'
# ENVIRONMENT_NAME = 'HalfCheetah-v1'
# ENVIRONMENT_NAME = 'Reacher-v2'

# Box2d
# ENVIRONMENT_NAME = 'Pendulum-v0'

# ENVIRONMENT_NAME = 'PongNoFrameskip-v4'
# ENVIRONMENT_NAME = 'CartPole-v1'
# ENVIRONMENT_NAME = 'Acrobot-v1'

# ENVIRONMENT_NAME = 'LunarLander-v2'
# ENVIRONMENT_NAME  ='LunarLanderContinuous-v2'
# (coord_x, coord_y, vel_x, vel_y, angle, angular_vel, l_leg_on_ground,
# r_leg_on_ground)

# ENVIRONMENT_NAME = 'MountainCar-v0'
# ENVIRONMENT_NAME = 'MountainCarContinuous-v0'

# ENVIRONMENT_NAME  ='CarRacing-v0'
# ENVIRONMENT_NAME  = 'BipedalWalkerHardcore-v2'
# ENVIRONMENT_NAME = 'BipedalWalker-v2'

MODEL_DIRECTORY = PROJECT_APP_PATH.user_data / ENVIRONMENT_NAME / LOAD_TIME / 'models'
CONFIG_DIRECTORY = PROJECT_APP_PATH.user_data / ENVIRONMENT_NAME / LOAD_TIME / 'configs'
LOG_DIRECTORY = PROJECT_APP_PATH.user_log / ENVIRONMENT_NAME / LOAD_TIME

# Architecture
POLICY_ARCH_SPEC = GDCS(CategoricalMLP,
                        NOD(input_shape=None,  # Obtain from environment
                            hidden_layers=(32, 32),  # Estimate from input and output size
                            output_shape=None,  # Obtain from environment
                            hidden_layer_activation=torch.relu,
                            use_bias=True
                            )
                        )

ROLLOUTS = 10000
