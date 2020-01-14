#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from logging import warning
from typing import Any, Sequence, Union

import torch
from tqdm import tqdm

from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter, TensorBoardPytorchWriter
from draugr.writers.terminal import sprint
from draugr.writers.writer import Writer
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import TrajectoryBuffer
from neodroidagent.common.architectures.architecture import Architecture
from neodroidagent.common.architectures.distributional.categorical import CategoricalMLP
from neodroidagent.common.architectures.distributional.normal import (
    MultiDimensionalNormalMLP,
)
from neodroidagent.common.architectures.mock import MockArchitecture
from neodroidagent.utilities.exceptions.exceptions import NoTrajectoryException
from neodroidagent.utilities.signal.discounting import discount_signal
from warg import drop_unused_kws
from warg.gdkc import GDKC
from warg.kw_passing import super_init_pass_on_kws

tqdm.monitor_interval = 0

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/09/2019
           """

__all__ = ["PGAgent"]


@super_init_pass_on_kws
class PGAgent(TorchAgent):
    r"""
    REINFORCE, Vanilla Policy Gradient method

  """

    def __init__(
        self,
        evaluation_function=torch.nn.CrossEntropyLoss(),
        policy_arch_spec=GDKC(
            CategoricalMLP,
            input_shape=None,
            # Obtain from environment
            hidden_layers=None,
            output_shape=None,
            # Obtain from environment
        ),
        discount_factor=0.95,
        optimiser_spec=GDKC(torch.optim.Adam, lr=2e-2),
        scheduler_spec=GDKC(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.65),
        state_type=torch.float,
        signals_tensor_type=torch.float,
        distribution_regressor: Architecture = MockArchitecture(),
        **kwargs,
    ) -> None:
        """

:param evaluation_function:
:param trajectory_trace:
:param policy_arch_spec:
:param discount_factor:
:param use_batched_updates:

:param signal_clipping:
:param signal_clip_high:
:param signal_clip_low:
:param optimiser_spec:
:param state_type:
:param signals_tensor_type:
:param grad_clip:
:param grad_clip_low:
:param grad_clip_high:
:param std:
:param distribution_regressor:
:param kwargs:
"""
        super().__init__(**kwargs)

        self._accumulated_error = to_tensor(0.0, device=self._device)
        self._trajectory_trace = TrajectoryBuffer()

        self._evaluation_function = evaluation_function
        self._policy_arch_spec = policy_arch_spec
        self._discount_factor = discount_factor

        self._optimiser_spec = optimiser_spec
        self._scheduler_spec = scheduler_spec
        self._state_type = state_type
        self._signals_tensor_type = signals_tensor_type

        self._distribution_regressor = distribution_regressor

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Writer = MockWriter(),
        print_model_repr: bool = True,
    ) -> None:

        self._policy_arch_spec.kwargs["input_shape"] = self._input_shape
        if action_space.is_discrete:
            self._policy_arch_spec = GDKC(CategoricalMLP, self._policy_arch_spec.kwargs)
            self._policy_arch_spec.kwargs["output_shape"] = self._output_shape
        else:
            self._policy_arch_spec = GDKC(
                MultiDimensionalNormalMLP, self._policy_arch_spec.kwargs
            )
            self._policy_arch_spec.kwargs["output_shape"] = self._output_shape

        self._distribution_regressor = self._policy_arch_spec().to(self._device)

        self._optimiser = self._optimiser_spec(
            self._distribution_regressor.parameters()
        )
        if self._scheduler_spec:
            self._scheduler = self._scheduler_spec(self._optimiser)

    @property
    def models(self) -> dict:
        return {"_distribution_regressor": self._distribution_regressor}

    @drop_unused_kws
    def _sample(self, state: Sequence) -> tuple:
        model_input = to_tensor(state, device=self._device, dtype=self._state_type)

        distributions = self._distribution_regressor(model_input)
        action, log_prob = self._distribution_regressor.sample(distributions)

        entropy = self._distribution_regressor.entropy(distributions)

        return action, log_prob, entropy

    @drop_unused_kws
    def _remember(self, *, signal, action_log_prob, entropy) -> None:
        self._trajectory_trace.add_point(signal, action_log_prob, entropy)

    # region Protected

    @drop_unused_kws
    def _update(self, *, metric_writer=MockWriter()) -> None:
        """

    :param metric_writer:
    :param args:
    :param kwargs:

    :returns:
    """

        self._optimiser.zero_grad()
        loss = self.evaluate()

        loss.backward()

        if self._gradient_clipping:
            for params in self._distribution_regressor.parameters():
                params.grad.data.clamp_(self._grad_clip_low, self._grad_clip_high)
        self._optimiser.step()

        if metric_writer:
            metric_writer.scalar("Loss", loss.detach().to("cpu").numpy())

        if self._scheduler:
            self._scheduler.step()
            if metric_writer:
                for i, param_group in enumerate(self._optimiser.param_groups):
                    metric_writer.scalar(f"lr{i}", param_group["lr"])

    # endregion

    # region Public

    @drop_unused_kws
    def evaluate(self) -> Any:
        if not len(self._trajectory_trace) > 0:
            raise NoTrajectoryException

        trajectory = self._trajectory_trace.retrieve_trajectory()
        t_signals = trajectory.signal
        log_probs = trajectory.log_prob
        self._trajectory_trace.clear()

        policy_loss = []

        signals = discount_signal(t_signals, self._discount_factor)

        signals = to_tensor(
            signals, device=self._device, dtype=self._signals_tensor_type
        )

        if signals.shape[0] > 1:
            signals = (signals - signals.mean()) / (
                signals.std() + self._divide_by_zero_safety
            )
        elif signals.shape[0] == 0:
            warning(f"No signals received, got signals.shape[0]: {signals.shape[0]}")

        for log_prob, signal in zip(log_probs, signals):
            policy_loss.append(-log_prob * signal)

        return torch.cat(policy_loss, 0).sum()

    # endregion


# region Test


def pg_run(
    rollouts=None, skip: bool = True, environment_type: Union[bool, str] = True
) -> None:
    from neodroidagent.common.sessions.session_entry_point import session_entry_point
    from neodroidagent.common.sessions.single_agent.parallel import ParallelSession
    from . import pg_test_config as C

    if rollouts:
        C.ROLLOUTS = rollouts

    session_entry_point(
        PGAgent,
        C,
        session=ParallelSession,
        skip_confirmation=skip,
        environment_type=environment_type,
    )


def pg_test() -> None:
    pg_run(environment_type="gym")


if __name__ == "__main__":
    pg_test()

# endregion
