#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import logging
import math
import random
from typing import Any, Dict, Sequence

import numpy
import torch
from torch.nn.functional import smooth_l1_loss

from draugr import MockWriter, Writer, to_tensor
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents import TorchAgent
from neodroidagent.common import (
    Architecture,
    SingleHeadMLP,
    Transition,
    TransitionPointBuffer,
    TransitionPoint,
)
from neodroidagent.utilities import (
    ActionSpaceNotSupported,
    ExplorationSpecification,
    update_target,
)
from neodroidagent.utilities.misc.bool_tests import is_zero_or_mod_zero
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"

__all__ = ["DQNAgent"]


@super_init_pass_on_kws(super_base=TorchAgent)
class DQNAgent(TorchAgent):
    """
Deep Q Network Agent
"""

    # region Private

    def __init__(
        self,
        value_arch_spec: Architecture = GDKC(
            SingleHeadMLP,
            input_shape=None,  # Obtain from environment
            hidden_layers=None,
            output_shape=None,  # Obtain from environment
        ),
        exploration_spec=ExplorationSpecification(start=0.9, end=0.05, decay=200),
        memory_buffer=TransitionPointBuffer(int(1e5)),
        # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
        loss_function=smooth_l1_loss,  # huber_loss
        batch_size=64,
        discount_factor=0.99,
        double_dqn=True,
        optimiser_spec=GDKC(torch.optim.Adam),
        scheduler_spec=None,
        sync_target_model_frequency=100,
        initial_observation_period=100,
        learning_frequency=4,
        update_target_percentage=1e-3,
        **kwargs,
    ):
        """

:param exploration_spec:
:param initial_observation_period:
:param value_model:
:param target_value_model:
:param naive_max_policy:
:param memory_buffer:
:param loss_function:
:param value_arch_spec:
:param batch_size:
:param discount_factor:
:param learning_frequency:
:param sync_target_model_frequency:
:param state_type:
:param value_type:
:param action_type:
:param double_dqn:
:param clamp_gradient:
:param signal_clipping:

:param optimiser_spec:
:param kwargs:
"""
        super().__init__(**kwargs)
        self._exploration_spec = exploration_spec
        self._memory_buffer = memory_buffer
        self._loss_function = loss_function
        self._value_arch_spec: Architecture = value_arch_spec
        self._batch_size = batch_size
        self._discount_factor = discount_factor

        self._state_type = torch.float
        self._value_type = torch.float
        self._action_type = torch.long

        self._double_dqn = double_dqn

        self._optimiser_spec = optimiser_spec
        self._scheduler_spec = scheduler_spec
        self._learning_frequency = learning_frequency
        self._initial_observation_period = initial_observation_period
        self._sync_target_model_frequency = sync_target_model_frequency
        self._update_target_percentage = update_target_percentage

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        writer: Writer = None,
        print_model_repr=True,
    ):
        """

@param observation_space:
@param action_space:
@param signal_space:
@param writer:
@param print_model_repr:
@return:
"""

        if action_space.is_continuous:
            raise ActionSpaceNotSupported

        self._value_arch_spec.kwargs["input_shape"] = self._input_shape
        self._value_arch_spec.kwargs["output_shape"] = self._output_shape

        self.value_model = self._value_arch_spec().to(self._device)
        self._target_value_model = (
            copy.deepcopy(self.value_model).to(self._device).eval()
        )

        self._optimiser = self._optimiser_spec(self.value_model.parameters())
        if self._scheduler_spec:
            self._scheduler = self._scheduler_spec(self._optimiser)
        else:
            self._scheduler = None

    # endregion

    # region Public
    @property
    def models(self) -> Dict[str, Architecture]:
        """

@return:
"""
        return {"value_model": self.value_model}

    def _exploration_sample(self, steps_taken, metric_writer=None):
        """
:param steps_taken:
:return:
"""
        assert 0 <= self._exploration_spec.end <= self._exploration_spec.start
        assert 0 < self._exploration_spec.decay

        if steps_taken == 0:
            return True

        sample = random.random()

        a = self._exploration_spec.start - self._exploration_spec.end

        b = math.exp(
            -1.0
            * steps_taken
            / (self._exploration_spec.decay + self._divide_by_zero_safety)
        )
        _current_eps_threshold = self._exploration_spec.end + a * b

        if metric_writer:
            metric_writer.scalar("Current Eps Threshold", _current_eps_threshold)

        return not sample > _current_eps_threshold

    # endregion

    # region Protected

    @drop_unused_kws
    def _sample(
        self,
        state: Sequence,
        deterministic: bool = False,
        metric_writer: Writer = MockWriter(),
    ) -> numpy.ndarray:
        """

@param state:
@param deterministic:
@param metric_writer:
@return:
"""
        if not deterministic and self._exploration_sample(
            self._sample_i, metric_writer
        ):
            return self._sample_random_process(state)

        return self._sample_model(state)

    def _sample_random_process(self, state) -> numpy.ndarray:
        r = numpy.arange(self._output_shape[0])
        sample = numpy.random.choice(r, len(state))
        return sample

    # endregion

    @drop_unused_kws
    def _remember(self, *, signal, terminated, transition):
        """

@param state:
@param action:
@param signal:
@param next_state:
@param terminated:
@return:
"""
        self._memory_buffer.add_transition_points(
            TransitionPoint(*transition, signal, terminated)
        )

    # region Protected

    @drop_unused_kws
    def _sample_model(self, state) -> numpy.ndarray:
        """

@param state:
@return:
"""
        model_input = to_tensor(state, device=self._device, dtype=self._state_type)

        with torch.no_grad():
            max_q_action = self.value_model(model_input).max(-1)[-1]

            return max_q_action.detach().to("cpu").numpy()

    @drop_unused_kws
    def _update(self, *, metric_writer=MockWriter()):
        """

@param metric_writer:
@return:
"""

        loss = math.inf

        if self._initial_observation_period < self.update_i:
            if is_zero_or_mod_zero(self._learning_frequency, self.update_i):
                if self._batch_size < len(self._memory_buffer):
                    transitions = self._memory_buffer.sample_transition_points(
                        self._batch_size
                    )

                    tensorised = TransitionPoint(
                        *[
                            to_tensor(a, device=self._device, dtype=d)
                            for a, d in zip(
                                transitions,
                                [torch.float, torch.long] + [torch.float] * 3,
                            )
                        ]
                    )

                    # Calculate Q of state
                    Q_state = self.value_model(tensorised.state).gather(
                        -1, tensorised.action.unsqueeze(-1)
                    )

                    # Calculate Q of successors
                    with torch.no_grad():
                        Q_successors = self.value_model(
                            tensorised.successor_state
                        ).detach()
                    successors_max_action = Q_successors.max(-1)[-1]
                    if self._double_dqn:
                        with torch.no_grad():
                            Q_successors = self._target_value_model(
                                tensorised.successor_state
                            ).detach()

                    max_next_values = Q_successors.gather(
                        -1, successors_max_action.unsqueeze(-1)
                    )

                    # Integrate with the true signal
                    Q_expected = tensorised.signal + (
                        self._discount_factor
                        * max_next_values
                        * tensorised.non_terminal_numerical
                    )

                    td_error = self._loss_function(Q_state, Q_expected)

                    self._optimiser.zero_grad()
                    td_error.backward()
                    self.post_process_gradients(self.value_model)
                    self._optimiser.step()

                    loss = td_error.mean().cpu().item()
                    if metric_writer:
                        metric_writer.scalar("td_error", loss, self.update_i)

                    if self._scheduler:
                        self._scheduler.step()
                        if metric_writer:
                            for i, param_group in enumerate(
                                self._optimiser.param_groups
                            ):
                                metric_writer.scalar(
                                    f"lr{i}", param_group["lr"], self.update_i
                                )

                else:
                    logging.info(
                        "Batch size is larger than current memory size, skipping update"
                    )

            if self._double_dqn:
                if is_zero_or_mod_zero(
                    self._sync_target_model_frequency, self.update_i
                ):
                    update_target(
                        target_model=self._target_value_model,
                        source_model=self.value_model,
                        copy_percentage=self._update_target_percentage,
                    )
                if metric_writer:
                    metric_writer.blip("Target Model Synced", self.update_i)

        return loss

    # endregion
