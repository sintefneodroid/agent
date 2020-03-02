#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import logging
import math
import random
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy
import torch
from torch.nn.functional import smooth_l1_loss

from draugr import MockWriter, Writer, to_tensor
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    Architecture,
    DuelingQMLP,
    TransitionPoint,
    TransitionPointPrioritisedBuffer,
    Memory,
)
from neodroidagent.utilities import (
    ActionSpaceNotSupported,
    ExplorationSpecification,
    is_zero_or_mod_zero,
    update_target,
)
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["DQNAgent"]


@super_init_pass_on_kws(super_base=TorchAgent)
class DQNAgent(TorchAgent):
    """
Deep Q Network Agent

https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
"""

    def __init__(
        self,
        value_arch_spec: Architecture = GDKC(DuelingQMLP),
        exploration_spec: GDKC = ExplorationSpecification(
            start=0.95, end=0.05, decay=3000
        ),
        memory_buffer: Memory = TransitionPointPrioritisedBuffer(int(1e5)),
        batch_size: int = 256,
        discount_factor: float = 0.95,
        double_dqn: bool = True,
        use_per: bool = True,
        loss_function: callable = smooth_l1_loss,
        optimiser_spec: GDKC = GDKC(torch.optim.Adam, lr=1e-4),
        scheduler_spec: GDKC = None,
        sync_target_model_frequency: int = 1,
        initial_observation_period: int = 1000,
        learning_frequency: int = 1,
        copy_percentage: float = 1e-2,
        **kwargs,
    ):
        """
@param value_arch_spec:
@param exploration_spec:
@param memory_buffer:
@param batch_size:
@param discount_factor:
@param double_dqn: https://arxiv.org/abs/1509.06461
@param use_per:  https://arxiv.org/abs/1511.05952
@param loss_function:  default is huber loss
@param optimiser_spec:
@param scheduler_spec:
@param sync_target_model_frequency:
@param initial_observation_period:
@param learning_frequency:
@param copy_percentage:
@param kwargs:
"""
        super().__init__(**kwargs)

        self._exploration_spec = exploration_spec
        assert 0 <= self._exploration_spec.end <= self._exploration_spec.start
        assert 0 < self._exploration_spec.decay

        self._memory_buffer = memory_buffer
        assert self._memory_buffer.capacity > batch_size

        self._value_arch_spec: Architecture = value_arch_spec
        self._optimiser_spec = optimiser_spec
        self._scheduler_spec = scheduler_spec

        self._batch_size = batch_size
        assert batch_size > 0

        self._discount_factor = discount_factor
        assert 0 <= discount_factor <= 1.0

        self._double_dqn = double_dqn
        self._use_per = use_per and double_dqn
        self._loss_function = loss_function

        self._learning_frequency = learning_frequency
        self._sync_target_model_frequency = sync_target_model_frequency

        self._initial_observation_period = initial_observation_period
        assert initial_observation_period >= 0

        self._copy_percentage = copy_percentage
        assert 0 <= copy_percentage <= 1.0

        self._state_type = torch.float
        self._value_type = torch.float
        self._action_type = torch.long

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

        if steps_taken == 0:
            return True

        a = self._exploration_spec.start - self._exploration_spec.end
        b = math.exp(
            -1.0
            * steps_taken
            / (self._exploration_spec.decay + self._divide_by_zero_safety)
        )
        _current_eps_threshold = self._exploration_spec.end + a * b

        if metric_writer:
            metric_writer.scalar("Current Eps Threshold", _current_eps_threshold)

        return not random.random() > _current_eps_threshold

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
        sample = numpy.random.choice(r, (len(state), 1))
        return sample

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

        a = [TransitionPoint(*s) for s in zip(*transition, signal, terminated)]
        if self._use_per:
            with torch.no_grad():
                td_error, *_ = self._td_error(zip(*a))
                for a_, e_ in zip(a, td_error.detach().squeeze(-1).cpu().numpy()):
                    self._memory_buffer.add_transition_point(a_, e_)
        else:
            for a_ in a:
                self._memory_buffer.add_transition_point(a_)

    @drop_unused_kws
    def _sample_model(self, state: Any) -> numpy.ndarray:
        """

@param state:
@return:
"""
        with torch.no_grad():
            max_q_action = self.value_model(
                to_tensor(state, device=self._device, dtype=self._state_type)
            )
            return max_q_action.max(-1)[-1].unsqueeze(-1).detach().to("cpu").numpy()

    def _max_q_successor(self, successor_state: torch.Tensor) -> torch.tensor:
        """

@param successor_state:
@return:
"""
        with torch.no_grad():
            Q_successors = self.value_model(successor_state).detach()

        successors_max_action = Q_successors.max(-1)[-1].unsqueeze(-1)

        if self._double_dqn:
            with torch.no_grad():
                Q_successors = self._target_value_model(successor_state).detach()

        return Q_successors.gather(-1, successors_max_action)

    def _q_expected(
        self, signal: torch.tensor, mask: torch.tensor, successor_state: torch.tensor
    ) -> torch.tensor:
        return (
            signal
            + self._discount_factor * self._max_q_successor(successor_state) * mask
        )

    def _q_state(self, state: torch.tensor, action: torch.tensor) -> torch.tensor:
        return self.value_model(state).gather(-1, action)

    def _td_error(self, transitions: Iterable) -> Tuple[torch.tensor, ...]:
        tensorised = TransitionPoint(
            *[
                to_tensor(a, device=self._device, dtype=d)
                for a, d in zip(
                    transitions, [torch.float, torch.long] + [torch.float] * 3
                )
            ]
        )

        Q_expected = self._q_expected(
            tensorised.signal,
            tensorised.non_terminal_numerical,
            tensorised.successor_state,
        )
        Q_state = self._q_state(tensorised.state, tensorised.action)

        return Q_expected - Q_state, Q_expected, Q_state

    @drop_unused_kws
    def _update(self, *, metric_writer: Writer = MockWriter()) -> None:
        """

@param metric_writer:
@return:
"""

        loss_ = math.inf

        if self.update_i > self._initial_observation_period:
            if is_zero_or_mod_zero(self._learning_frequency, self.update_i):
                if len(self._memory_buffer) > self._batch_size:
                    transitions = self._memory_buffer.sample(self._batch_size)

                    td_error, Q_expected, Q_state = self._td_error(transitions)
                    td_error = td_error.detach().squeeze(-1).cpu().numpy()

                    if self._use_per:
                        self._memory_buffer.update_last_batch(td_error)

                    loss = self._loss_function(Q_state, Q_expected)

                    self._optimiser.zero_grad()
                    loss.backward()
                    self.post_process_gradients(self.value_model.parameters())
                    self._optimiser.step()

                    loss_ = loss.detach().cpu().item()
                    if metric_writer:
                        metric_writer.scalar("td_error", td_error.mean(), self.update_i)
                        metric_writer.scalar("loss", loss_, self.update_i)

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
                        copy_percentage=self._copy_percentage,
                    )
                if metric_writer:
                    metric_writer.blip("Target Model Synced", self.update_i)

        return loss_
