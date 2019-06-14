#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = '''
Description: Config for training
Author: Christian Heider Nielsen
'''

# noinspection PyUnresolvedReferences
from agent.configs.base_config import *

CONFIG_NAME = __name__
CONFIG_FILE = __file__

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

ENVIRONMENT_NAME = 'CartPole-v1'
'''

Description
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum 
starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's 
velocity.

Source
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

Observation
Type: Box(4)

Num	Observation	Min	Max
0	Cart Position	-2.4	2.4
1	Cart Velocity	-Inf	Inf
2	Pole Angle	~ -41.8°	~ 41.8°
3	Pole Velocity At Tip	-Inf	Inf
Actions
Type: Discrete(2)

Num	Action
0	Push cart to the left
1	Push cart to the right

Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is 
pointing. This is because the center of gravity of the pole increases the amount of energy needed to move 
the cart underneath it

Reward
Reward is 1 for every step taken, including the termination step

Starting State
All observations are assigned a uniform random value between ±0.05

Episode Termination
Pole Angle is more than ±12°
Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
Episode length is greater than 200
Solved Requirements
Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.

'''
