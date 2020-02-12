#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Sequence

import numpy
import torch
from torch.nn import Module
from torch.optim import Optimizer
from tqdm import tqdm

from draugr import MockWriter, Writer, to_tensor
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroid.utilities.transformations.terminal_masking import (
    non_terminal_numerical_mask,
)
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    CategoricalMLP,
    MultiDimensionalNormalMLP,
    SamplePoint,
    SampleTrajectoryBuffer,
)
from neodroidagent.utilities import NoTrajectoryException, discount_signal_torch
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

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
        policy_arch_spec=GDKC(CategoricalMLP),
        discount_factor=0.95,
        optimiser_spec=GDKC(torch.optim.Adam, lr=1e-4),
        scheduler_spec=GDKC(torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.65),
        memory_buffer=SampleTrajectoryBuffer(),
        **kwargs,
    ) -> None:
        r"""
    :param evaluation_function:
    :param trajectory_trace:
    :param policy_arch_spec:
    :param discount_factor:
    :param optimiser_spec:
    :param state_type:
    :param kwargs:
    """
        super().__init__(**kwargs)

        self._memory_buffer = memory_buffer

        self._evaluation_function = evaluation_function
        self._policy_arch_spec = policy_arch_spec
        self._discount_factor = discount_factor

        self._optimiser_spec = optimiser_spec
        self._scheduler_spec = scheduler_spec

        self._mask_terminated_signals = False

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Writer = MockWriter(),
        print_model_repr: bool = True,
        *,
        distributional_regressor: Module = None,
        optimiser: Optimizer = None,
    ) -> None:
        """

    @param observation_space:
    @param action_space:
    @param signal_space:
    @param metric_writer:
    @param print_model_repr:
    @param distributional_regressor:
    @param optimiser:
    @return:
    """

        if distributional_regressor:
            self.distributional_regressor = distributional_regressor
        else:
            self._policy_arch_spec.kwargs["input_shape"] = self._input_shape
            if action_space.is_discrete:
                self._policy_arch_spec = GDKC(
                    CategoricalMLP, self._policy_arch_spec.kwargs
                )
            else:
                self._policy_arch_spec = GDKC(
                    MultiDimensionalNormalMLP, self._policy_arch_spec.kwargs
                )

            self._policy_arch_spec.kwargs["output_shape"] = self._output_shape

            self.distributional_regressor = self._policy_arch_spec().to(self._device)

        if optimiser:
            self._optimiser = optimiser
        else:
            self._optimiser = self._optimiser_spec(
                self.distributional_regressor.parameters()
            )

        if self._scheduler_spec:
            self._scheduler = self._scheduler_spec(self._optimiser)
        else:
            self._scheduler = None

    @property
    def models(self) -> dict:
        """

    @return:
    """
        return {"distributional_regressor": self.distributional_regressor}

    @drop_unused_kws
    def _sample(self, state: Sequence) -> SamplePoint:
        """

    @param state:
    @return:
    """
        model_input = to_tensor(state, device=self._device, dtype=torch.float)
        distribution = self.distributional_regressor(model_input)

        with torch.no_grad():
            action = distribution.sample().detach()
        return SamplePoint(action, distribution)

    def extract_action(self, sample: SamplePoint) -> numpy.ndarray:
        action, _ = sample

        return action.to("cpu").numpy()

    @drop_unused_kws
    def _remember(self, *, signal: Any, terminated: Any, sample: SamplePoint) -> None:
        """

    @param signal:
    @param terminated:
    @param sample:
    @return:
    """

        self._memory_buffer.add_trajectory_point(signal, terminated, *sample)

    # region Protected

    @drop_unused_kws
    def _update(self, *, metric_writer=MockWriter()) -> float:
        """

:param metric_writer:

:returns:
"""

        loss = self.evaluate()

        self._optimiser.zero_grad()
        loss.backward()
        self.post_process_gradients(self.distributional_regressor)
        self._optimiser.step()

        if metric_writer:
            metric_writer.scalar("Loss", loss.detach().to("cpu").numpy())

        if self._scheduler:
            self._scheduler.step()
            if metric_writer:
                for i, param_group in enumerate(self._optimiser.param_groups):
                    metric_writer.scalar(f"lr{i}", param_group["lr"])

        return loss.cpu().item()

    # endregion

    # region Public

    @drop_unused_kws
    def evaluate(self) -> Any:
        """

    @return:
    """
        if not len(self._memory_buffer) > 0:
            raise NoTrajectoryException

        trajectory = self._memory_buffer.retrieve_trajectory()
        self._memory_buffer.clear()

        log_probs = to_tensor(
            [d.log_prob(a) for d, a in zip(trajectory.distribution, trajectory.action)],
            device=self._device,
        )

        signal = (
            to_tensor(trajectory.signal, device=self._device).squeeze(-1).T.squeeze(-1)
        )
        non_terminal = (
            to_tensor(
                non_terminal_numerical_mask(trajectory.terminated), device=self._device
            )
            .squeeze(-1)
            .T.squeeze(-1)
        )

        discounted_signal = discount_signal_torch(
            signal,
            self._discount_factor,
            device=self._device,
            non_terminal=non_terminal,
        ).T

        discounted_signal = (discounted_signal - discounted_signal.mean()) / (
            discounted_signal.std() + self._divide_by_zero_safety
        )

        return -(log_probs * discounted_signal).mean()

    # endregion
