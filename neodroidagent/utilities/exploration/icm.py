#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy
import torch
from neodroid.utilities import ActionSpace, ObservationSpace
from torch import nn
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss, MSELoss

from draugr.torch_utilities.to_tensor import to_tensor
from neodroidagent.utilities.exploration.curiosity_module import CuriosityModule

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """
__all__ = ["ForwardModel", "InverseModel", "MLPICM"]


class ForwardModel(nn.Module):
    """

"""

    def __init__(self, action_converter: ActionSpace, state_latent_features: int):
        """

    @param action_converter:
    @param state_latent_features:
    """
        super().__init__()

        action_latent_features = 128
        if action_converter.is_discrete:
            self.action_encoder = nn.Embedding(
                action_converter.shape[0], action_latent_features
            )
        else:
            self.action_encoder = nn.Linear(
                action_converter.shape[0], action_latent_features
            )
        self.hidden = nn.Sequential(
            nn.Linear(action_latent_features + state_latent_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, state_latent_features),
        )

    def forward(self, state_latent: torch.Tensor, action: torch.Tensor):
        """

    @param state_latent:
    @param action:
    @return:
    """
        action = self.action_encoder(
            action.long() if self.action_converter.discrete else action
        )
        x = torch.cat((action, state_latent), dim=-1)
        x = self.hidden(x)
        return x


class InverseModel(nn.Module):
    """

"""

    def __init__(self, action_space: ActionSpace, state_latent_features: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_space.n),
        )

    def forward(self, state_latent: torch.Tensor, next_state_latent: torch.Tensor):
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))


class MLPICM(CuriosityModule):
    """
Implements the Intrinsic Curiosity Module described in paper: https://arxiv.org/pdf/1705.05363.pdf

The overview of the idea is to reward the agent for exploring unseen states. It is achieved by
implementing two models. One called forward model that given the encoded state and encoded action
computes predicts the encoded next state. The other one called inverse model that given the encoded state
and encoded next_state predicts action that
must have been taken to move from one state to the other. The final intrinsic reward is the difference
between encoded next state and encoded next state predicted by the forward module. Inverse model is there
to make sure agent focuses on the states that he actually can control.
"""

    def __init__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        policy_weight: float,
        reward_scale: float,
        weight: float,
        intrinsic_reward_integration: float,
    ):
        """
:param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to
control how
   important optimizing policy to optimizing the curiosity module
:param reward_scale: scales the intrinsic reward returned by this module. Can be used to control how
big the
   intrinsic reward is
:param weight: balances the importance between forward and inverse model
:param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic reward.
Used when
   incorporating intrinsic into extrinsic in the ``reward`` method
"""

        assert (
            len(observation_space.shape) == 1
        ), "Only flat spaces supported by MLP model"
        assert (
            len(action_space.shape) == 1
        ), "Only flat action spaces supported by MLP model"
        super().__init__()

        self.input_state_shape = observation_space
        self.input_action_shape = action_space
        self.policy_weight = policy_weight
        self.reward_scale = reward_scale
        self.weight = weight
        self.intrinsic_reward_integration = intrinsic_reward_integration

        self.encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

        self.forward_model = ForwardModel(action_space, 128)
        self.inverse_model = InverseModel(action_space, 128)

        self.a_loss = CrossEntropyLoss()
        self.a_loss = MSELoss()

    def forward(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ):
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    def reward(
        self, rewards: numpy.ndarray, states: numpy.ndarray, actions: numpy.ndarray
    ) -> numpy.ndarray:
        n, t = actions.shape[0], actions.shape[1]
        states, next_states = states[:, :-1], states[:, 1:]
        states = to_tensor(
            states.reshape(states.shape[0] * states.shape[1], -1)
        )  # flatten
        next_states = to_tensor(
            next_states.reshape(states.shape[0] * states.shape[1], -1)
        )
        actions = to_tensor(actions.reshape(n * t, *actions.shape[2:]))
        next_states_latent, next_states_hat, _ = self.forward(
            states, next_states, actions
        )
        intrinsic_reward = (
            self.reward_scale
            / 2
            * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2)
        )
        intrinsic_reward = intrinsic_reward.cpu().detach().numpy().reshape(n, t)

        return (
            1.0 - self.intrinsic_reward_integration
        ) * rewards + self.intrinsic_reward_integration * intrinsic_reward

    #     self.reporter.scalar('icm/reward',
    #                          intrinsic_reward.mean().item()
    #                          if self.reporter.will_report('icm/reward')
    #                          else 0)

    def loss(
        self,
        policy_loss: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        next_states_latent, next_states_hat, actions_hat = self.forward(
            states, next_states, actions
        )
        forward_loss = (
            0.5
            * (next_states_hat - next_states_latent.detach())
            .norm(2, dim=-1)
            .pow(2)
            .mean()
        )
        ca = Categorical(logits=actions_hat).sample()
        inverse_loss = self.a_loss(ca, actions)
        curiosity_loss = self.weight * forward_loss + (1 - self.weight) * inverse_loss

        return self.policy_weight * policy_loss + curiosity_loss

        #    self.reporter.scalar('icm/loss', curiosity_loss.item())
