#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 10/12/2019
           """

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from draugr import global_torch_device, to_tensor
from neodroid.utilities import ActionSpace
from neodroidagent.common.architectures.architecture import Architecture

from warg import super_init_pass_on_kws


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


@super_init_pass_on_kws
class QNetwork(Architecture):
    """
The Q value network output an expected value given the state and action taken
"""

    def __init__(self, input_shape, output_shape, hidden_layers, **kwargs):
        super().__init__(**kwargs)

        i, o = input_shape[0], output_shape[0]
        h = hidden_layers[0]
        self.linear1 = nn.Linear(i + o, h)
        self.linear2 = nn.Linear(h, h)
        self.linear3 = nn.Linear(h, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        x = torch.cat((state, action), -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


@super_init_pass_on_kws
class PolicyNetwork(Architecture):
    """
The policy network has two outputs: The mean and the log standard deviation
"""

    def __init__(
        self,
        input_shape,
        output_shape,
        hidden_layers,
        init_w=3e-3,
        log_std_min=-20,
        log_std_max=2,
        action_space: ActionSpace = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        i, o = input_shape[0], output_shape[0]
        h = hidden_layers[0]

        self.linear1 = nn.Linear(i, h)
        self.linear2 = nn.Linear(h, h)

        self.mean_linear = nn.Linear(h, o)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(h, o)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

        self.normal_01 = Normal(0, 1)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = to_tensor(1.0)
            self.action_bias = to_tensor(0.0)
        else:
            self.action_scale = to_tensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = to_tensor((action_space.high + action_space.low) / 2.0)

    def to(self, device, **kwargs):
        """

@param device:
@param kwargs:
@return:
"""
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device, **kwargs)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample(
        self, state, epsilon=1e-6
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # action, z = self.reparameterised_sample(mean, std)
        normal = Normal(mean, std)
        action = normal.rsample()  # reparameterised_sample
        log_prob = normal.log_prob(action)
        action_act = torch.tanh(action) * self.action_scale + self.action_bias
        log_prob -= torch.log(self.action_scale * (1 - action_act.pow(2)) + epsilon)

        log_prob = log_prob.sum(-1, keepdim=True)

        eval_action_act = torch.tanh(mean) * self.action_scale + self.action_bias

        # print(action.shape, log_prob.shape, eval_action_act.shape)

        return action, log_prob, eval_action_act

    def reparameterised_sample(self, mean, std, device=global_torch_device()):
        """
#Reparameterisation trick to make sampling differientable

@param device:
@param mean:
@param std:
@return:
"""

        # sample randomly from a Standard Normal distribution
        z = self.normal_01.sample().to(device)

        # multiply it with our standard devation and add it to the mean
        action = mean + std * z

        return action, z


class DeterministicPolicy(Architecture):
    def __init__(
        self, num_inputs, num_actions, hidden_dim, action_space=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = to_tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.0
            self.action_bias = 0.0
        else:
            self.action_scale = to_tensor((action_space.high - action_space.low) / 2.0)
            self.action_bias = to_tensor((action_space.high + action_space.low) / 2.0)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0.0, std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.0), mean

    def to(self, device, **kwargs):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super().to(device, **kwargs)
