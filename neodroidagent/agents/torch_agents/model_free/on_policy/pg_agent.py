#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 23/09/2019
           """

__all__ = ["PolicyGradientAgent"]

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy
import torch
from draugr.torch_utilities import to_tensor, CategoricalMLP, MultiDimensionalNormalMLP
from draugr.writers import MockWriter, Writer
from neodroidagent.utilities.misc.common_metrics import CommonProcedureScalarEnum
from torch.nn import Module
from torch.optim import Optimizer
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

from neodroid.utilities import (
    non_terminal_numerical_mask,
)
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    Memory,
    SamplePoint,
    SampleTrajectoryBuffer,
)
from neodroidagent.utilities import NoTrajectoryException, discount_rollout_signal_torch
from trolls.spaces import (
    ActionSpace,
    ObservationSpace,
    SignalSpace,
)


@super_init_pass_on_kws
class PolicyGradientAgent(TorchAgent):
    r"""
    REINFORCE, Vanilla Policy Gradient method

    - Williams 1992 paper
    - Silver and Sutton update the policy in every timestep
    """

    def __init__(
        self,
        evaluation_function: callable = torch.nn.CrossEntropyLoss(),
        policy_arch_spec: GDKC = GDKC(
            constructor=CategoricalMLP, hidden_layer_activation=torch.nn.Tanh()
        ),
        discount_factor: float = 0.95,
        optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        scheduler_spec: GDKC = GDKC(
            constructor=torch.optim.lr_scheduler.StepLR, step_size=100, gamma=0.65
        ),
        memory_buffer: Memory = SampleTrajectoryBuffer(),
        **kwargs,
    ) -> None:
        r"""
        :param evaluation_function:
        :param trajectory_trace:
        :param policy_arch_spec:
        :param discount_factor:
        :param optimiser_spec:
        :param state_type:
        :param kwargs:"""
        super().__init__(**kwargs)

        assert 0 <= discount_factor <= 1.0

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
        metric_writer: Optional[Writer] = MockWriter(),
        print_model_repr: bool = True,
        *,
        distributional_regressor: Module = None,
        optimiser: Optimizer = None,
    ) -> None:
        """

        :param observation_space:
        :param action_space:
        :param signal_space:
        :param metric_writer:
        :param print_model_repr:
        :param distributional_regressor:
        :param optimiser:
        :return:"""

        if distributional_regressor:
            self.distributional_regressor = distributional_regressor
        else:
            self._policy_arch_spec.kwargs["input_shape"] = self._input_shape
            if action_space.is_singular_discrete:
                self._policy_arch_spec = GDKC(
                    constructor=CategoricalMLP, kwargs=self._policy_arch_spec.kwargs
                )
            else:
                self._policy_arch_spec = GDKC(
                    constructor=MultiDimensionalNormalMLP,
                    kwargs=self._policy_arch_spec.kwargs,
                )

            self._policy_arch_spec.kwargs["output_shape"] = self._output_shape

            self.distributional_regressor: Module = self._policy_arch_spec().to(
                self._device
            )

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

        :return:"""
        return {"distributional_regressor": self.distributional_regressor}

    @property
    def optimisers(self) -> Dict[str, Dict[str, Optimizer]]:
        return {"distributional_regressor": {"_optimiser": self._optimiser}}

    @drop_unused_kws
    def _sample(self, state: Sequence) -> Tuple:
        """

        :param state:
        :return:"""
        model_input = to_tensor(state, device=self._device, dtype=torch.float)
        distribution = self.distributional_regressor(model_input)

        with torch.no_grad():
            action = distribution.sample().detach()

        if self.action_space.is_singular_discrete:
            action = action.unsqueeze(-1)
        else:
            pass
            # action = torch.clamp(action, -1, 1)

        return action, distribution

    def extract_action(self, sample) -> numpy.ndarray:
        return sample[0].to("cpu").numpy()

    @drop_unused_kws
    def _remember(self, *, signal: Any, terminated: Any, sample: SamplePoint) -> None:
        """

        :param signal:
        :param terminated:
        :param sample:
        :return:"""
        action, dist = sample
        self._memory_buffer.add_trajectory_point(signal, terminated, action, dist)

    # region Protected

    def get_log_prob(self, dist, action):
        if self.action_space.is_singular_discrete:
            return dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        else:
            return dist.log_prob(action).sum(axis=-1, keepdims=True)

    @drop_unused_kws
    def _update(self, *, metric_writer=MockWriter()) -> float:
        """

        :param metric_writer:

        :returns:"""

        if not len(self._memory_buffer) > 0:
            raise NoTrajectoryException

        trajectory = self._memory_buffer.retrieve_trajectory()
        self._memory_buffer.clear()

        log_probs = to_tensor(
            [
                self.get_log_prob(d, a)
                for d, a in zip(trajectory.distribution, trajectory.action)
            ],
            device=self._device,
        )

        signal = to_tensor(trajectory.signal, device=self._device)
        non_terminal = to_tensor(
            non_terminal_numerical_mask(trajectory.terminated), device=self._device
        )

        discounted_signal = discount_rollout_signal_torch(
            signal,
            self._discount_factor,
            device=self._device,
            non_terminal=non_terminal,
        )

        loss = -(log_probs * discounted_signal).mean()

        self._optimiser.zero_grad()
        loss.backward()
        self.post_process_gradients(
            self.distributional_regressor.parameters(),
            metric_writer=metric_writer,
            parameter_set_name="distributional_regressor_model_parameters",
        )
        self._optimiser.step()

        if self._scheduler:
            self._scheduler.step()
            if metric_writer:
                for i, param_group in enumerate(self._optimiser.param_groups):
                    metric_writer.scalar(f"lr{i}", param_group["lr"])

        loss_cpu = loss.detach().to("cpu").numpy()
        if metric_writer:
            metric_writer.scalar(CommonProcedureScalarEnum.loss.value, loss_cpu)

        return loss_cpu.item()

    # endregion
