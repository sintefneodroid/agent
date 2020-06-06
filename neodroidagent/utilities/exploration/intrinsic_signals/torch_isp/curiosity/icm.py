#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Tuple

import numpy
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import CrossEntropyLoss, MSELoss

from draugr.torch_utilities.tensors.to_tensor import to_tensor
from draugr.writers import Writer
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
           """
__all__ = ["ForwardModel", "InverseModel", "MLPICM"]

from neodroidagent.utilities.exploration.intrinsic_signals.torch_isp.torch_isp_module import (
    TorchISPModule,
)


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

    def forward(self, state_latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        """

    @param action_space:
    @type action_space:
    @param state_latent_features:
    @type state_latent_features:
    """
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(state_latent_features * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_space.n),
        )

    def forward(
        self, state_latent: torch.Tensor, next_state_latent: torch.Tensor
    ) -> torch.Tensor:
        """

    @param state_latent:
    @type state_latent:
    @param next_state_latent:
    @type next_state_latent:
    @return:
    @rtype:
    """
        return self.input(torch.cat((state_latent, next_state_latent), dim=-1))


class MLPICM(TorchISPModule):
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
        signal_space: SignalSpace,
        policy_weight: float,
        weight: float,
        intrinsic_reward_integration: float,
        hidden_dim: int = 128,
    ):
        """
:param policy_weight: weight to be applied to the ``policy_loss`` in the ``loss`` method. Allows to
control how
important optimizing policy to optimizing the curiosity module
:param signal_space: used for scaling the intrinsic reward returned by this module. Can be used to control how
the fluctuation scale of the intrinsic signal
:param weight: balances the importance between forward and inverse model
:param intrinsic_reward_integration: balances the importance between extrinsic and intrinsic signal.
"""

        assert (
            len(observation_space.shape) == 1
        ), "Only flat spaces supported by MLP model"
        assert (
            len(action_space.shape) == 1
        ), "Only flat action spaces supported by MLP model"
        super().__init__(observation_space, action_space, signal_space)

        self.policy_weight = policy_weight
        self.reward_scale = signal_space.span
        self.weight = weight
        self.intrinsic_signal_integration = intrinsic_reward_integration

        self.encoder = nn.Sequential(
            nn.Linear(observation_space.shape[0], hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.forward_model = ForwardModel(action_space, hidden_dim)
        self.inverse_model = InverseModel(action_space, hidden_dim)

        self.a_loss = CrossEntropyLoss()
        self.a_loss = MSELoss()

    def forward(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple:
        """

    @param state:
    @type state:
    @param next_state:
    @type next_state:
    @param action:
    @type action:
    @return:
    @rtype:
    """
        state = self.encoder(state)
        next_state = self.encoder(next_state)
        next_state_hat = self.forward_model(state, action)
        action_hat = self.inverse_model(state, next_state)
        return next_state, next_state_hat, action_hat

    def sample(
        self,
        signals: numpy.ndarray,
        states: numpy.ndarray,
        actions: numpy.ndarray,
        *,
        writer: Writer = None
    ) -> numpy.ndarray:
        """

    @param signals:
    @type signals:
    @param states:
    @type states:
    @param actions:
    @type actions:
    @param writer:
    @type writer:
    @return:
    @rtype:
    """
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
        intrinsic_signal = (
            (
                self.reward_scale
                / 2
                * (next_states_hat - next_states_latent).norm(2, dim=-1).pow(2)
            )
            .cpu()
            .detach()
            .numpy()
            .reshape(n, t)
        )

        if writer is not None:
            writer.scalar("icm/signal", intrinsic_signal.mean().item())

        return (
            1.0 - self.intrinsic_signal_integration
        ) * signals + self.intrinsic_signal_integration * intrinsic_signal

    def loss(
        self,
        policy_loss: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
        *,
        writer: Writer = None
    ) -> torch.Tensor:
        """

    @param policy_loss:
    @type policy_loss:
    @param states:
    @type states:
    @param next_states:
    @type next_states:
    @param actions:
    @type actions:
    @param writer:
    @type writer:
    @return:
    @rtype:
    """
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

        if writer is not None:
            writer.scalar("icm/loss", curiosity_loss.item())

        return self.policy_weight * policy_loss + curiosity_loss
