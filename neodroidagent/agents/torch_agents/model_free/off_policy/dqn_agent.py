#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import logging
import math
import random
from typing import Dict, Sequence, Union

import numpy
import torch
from torch.nn.functional import smooth_l1_loss

from draugr.torch_utilities import copy_state, to_tensor
from draugr.writers import MockWriter, TensorBoardPytorchWriter, sprint, Writer
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace

from neodroidagent.common.architectures.architecture import Architecture
from neodroidagent.common.architectures.mlp import SingleHeadMLP
from neodroidagent.utilities.bool_tests import is_set_mod_zero_ret_alt

from neodroidagent.utilities.exceptions.exceptions import ActionSpaceNotSupported
from neodroidagent.common.memory import ReplayBuffer
from neodroidagent.common.procedures.training.off_policy_episodic import (
    OffPolicyEpisodic,
)

from neodroidagent.utilities import update_target
from neodroidagent.utilities.exploration.exploration_specification import (
    ExplorationSpecification,
)
from warg.gdkc import GDKC
from warg.kw_passing import drop_unused_kws, super_init_pass_on_kws
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent

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
        exploration_spec=ExplorationSpecification(0.995, 0.05, 10000),
        naive_max_policy=False,
        memory_buffer=ReplayBuffer(10000),
        # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
        loss_function=smooth_l1_loss,  # huber_loss
        batch_size=128,
        discount_factor=0.95,
        sync_target_model_frequency=1000,
        use_double_dqn=True,
        early_stopping_condition=None,
        optimiser_spec=GDKC(torch.optim.Adam),
        scheduler_spec=None,
        learning_frequency=1,
        initial_observation_period=1000,
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
    :param use_double_dqn:
    :param clamp_gradient:
    :param signal_clipping:
    :param early_stopping_condition:
    :param optimiser_spec:
    :param kwargs:
    """
        super().__init__(**kwargs)
        self._exploration_spec = exploration_spec
        self._naive_max_policy = naive_max_policy
        self._memory_buffer = memory_buffer
        self._loss_function = loss_function
        self._value_arch_spec: Architecture = value_arch_spec
        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._sync_target_model_frequency = sync_target_model_frequency
        self._state_type = torch.float
        self._value_type = torch.float
        self._action_type = torch.long

        self._use_double_dqn = use_double_dqn

        self._early_stopping_condition = early_stopping_condition
        self._optimiser_spec = optimiser_spec
        self._scheduler_spec = scheduler_spec
        self._learning_frequency = learning_frequency
        self._initial_observation_period = initial_observation_period

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
        self._value_model = self._value_arch_spec().to(self._device)

        self._target_value_model = self._value_arch_spec().to(self._device)
        self._target_value_model: Architecture = copy_state(
            target=self._target_value_model, source=self._value_model
        )
        self._target_value_model.eval()

        self._optimiser = self._optimiser_spec(self._value_model.parameters())
        if self._scheduler_spec:
            self._scheduler = self._scheduler_spec(self._optimiser)

        self._current_eps_threshold = 0

    # endregion

    # region Public
    @property
    def models(self) -> Dict[str, Architecture]:
        """

    @return:
    """
        return {"_value_model": self._value_model}

    def epsilon_random_exploration(self, steps_taken):
        """
    :param steps_taken:
    :return:
    """
        assert 0 <= self._exploration_spec.end <= self._exploration_spec.start

        if steps_taken == 0:
            return False

        sample = random.random()

        a = self._exploration_spec.start - self._exploration_spec.end

        b = math.exp(
            -1.0
            * steps_taken
            / (self._exploration_spec.decay + self._divide_by_zero_safety)
        )
        self._current_eps_threshold = self._exploration_spec.end + a * b

        return sample > self._current_eps_threshold

    # endregion

    # region Protected

    @drop_unused_kws
    def _sample(
        self, state: Sequence, no_random=False, metric_writer: Writer = MockWriter()
    ):
        """

    @param state:
    @param no_random:
    @param metric_writer:
    @return:
    """
        self._sample_i_since_update += 1
        s = self.epsilon_random_exploration(self._sample_i_since_update)
        if metric_writer:
            metric_writer.scalar(
                "Current Eps Threshold",
                self._current_eps_threshold,
                self._sample_i_since_update,
            )

        if (
            s and self._sample_i_since_update > self._initial_observation_period
        ) or no_random:
            return self._sample_model(state)

        return self._sample_random_process(state)

    def _sample_random_process(self, state):
        r = numpy.arange(self.output_shape[0])
        sample = numpy.random.choice(r, len(state))
        return sample

    # endregion

    @drop_unused_kws
    def _remember(self, *, state, action, signal, next_state, terminated):
        """

    @param state:
    @param action:
    @param signal:
    @param next_state:
    @param terminated:
    @return:
    """
        for a in zip(state, action, signal, next_state, terminated):
            self._memory_buffer.add_transition(*a)

    # region Protected

    @drop_unused_kws
    def _sample_model(self, state):
        """

    @param state:
    @return:
    """
        model_input = to_tensor(state, device=self._device, dtype=self._state_type)

        with torch.no_grad():
            action_value_estimates = self._value_model(model_input)
            max_value_action_idx = (
                action_value_estimates.max(-1)[1].to("cpu").numpy().tolist()
            )

        return max_value_action_idx

    @drop_unused_kws
    def _update(self, *, metric_writer=MockWriter()):
        """

    @param metric_writer:
    @return:
    """

        if not is_set_mod_zero_ret_alt(self._learning_frequency, self._update_i):
            return

        if self._batch_size < len(self._memory_buffer):
            transitions = self._memory_buffer.sample_transitions(self._batch_size)

            self._optimiser.zero_grad()
            td_error = self.evaluate(transitions)
            td_error.backward()
            if self._gradient_clipping:
                for params in self._value_model.parameters():
                    params.grad.data.clamp_(self._grad_clip_low, self._grad_clip_high)
            self._optimiser.step()

            if metric_writer:
                metric_writer.scalar("td_error", td_error.mean().item())

            if self._scheduler:
                self._scheduler.step()
                if metric_writer:
                    for i, param_group in enumerate(self._optimiser.param_groups):
                        metric_writer.scalar(f"lr{i}", param_group["lr"])

        else:
            logging.info(
                "Batch size is larger than current memory size, skipping update"
            )

        if (
            self._use_double_dqn
            and self._update_i % self._sync_target_model_frequency == 0
        ):
            update_target(
                target_model=self._target_value_model, source_model=self._value_model
            )
            if metric_writer:
                metric_writer.scalar(
                    "Target Model Synced", self._update_i, self._update_i
                )

    # endregion

    # region Public

    @drop_unused_kws
    def evaluate(self, batch):
        """

    :param batch:
    :type batch:
    :return:
    :rtype:
    """

        states = to_tensor(batch.state, dtype=self._state_type, device=self._device)

        true_signals = to_tensor(
            batch.signal, dtype=self._value_type, device=self._device
        )

        action_indices = to_tensor(
            batch.action, dtype=self._action_type, device=self._device
        )

        non_terminal_mask = to_tensor(
            batch.non_terminal_numerical, dtype=torch.float, device=self._device
        )

        successor_states = to_tensor(
            batch.successor_state, dtype=self._state_type, device=self._device
        )

        # Calculate Q of successors
        with torch.no_grad():
            Q_successors = self._value_model(successor_states)

        Q_successors_max_action_indices = Q_successors.max(-1)[1]
        Q_successors_max_action_indices = Q_successors_max_action_indices.unsqueeze(-1)
        if self._use_double_dqn:
            with torch.no_grad():
                Q_successors = self._target_value_model(successor_states)

        max_next_values = Q_successors.gather(
            -1, Q_successors_max_action_indices
        ).squeeze(-1)
        # a = Q_max_successor[non_terminal_mask]
        Q_max_successor = max_next_values * non_terminal_mask

        # Integrate with the true signal
        Q_expected = true_signals + (self._discount_factor * Q_max_successor)

        # Calculate Q of state
        action_indices = action_indices.unsqueeze(-1)
        p = self._value_model(states)
        Q_state = p.gather(-1, action_indices).squeeze(-1)

        return self._loss_function(Q_state, Q_expected)

    @drop_unused_kws
    def infer(self, state):
        model_input = to_tensor(state, device=self._device, dtype=self._state_type)
        with torch.no_grad():
            value = self._value_model(model_input)
        return value

    def step(self, state, env):
        action = self.sample(state)
        return action, env.react(action)

    # endregion


# region Test


def dqn_run(
    rollouts=None, skip: bool = True, environment_type: Union[bool, str] = True
):
    from neodroidagent.common.sessions.session_entry_point import session_entry_point
    from neodroidagent.common.sessions.single_agent.parallel import ParallelSession
    from . import dqn_test_config as C

    if rollouts:
        C.ROLLOUTS = rollouts

    session_entry_point(
        DQNAgent,
        C,
        session=ParallelSession(
            environment_name=C.ENVIRONMENT_NAME,
            procedure=OffPolicyEpisodic,
            # auto_reset_on_terminal_state=True,
            environment_type=environment_type,
        ),
        skip_confirmation=skip,
        environment_type=environment_type,
    )


def dqn_test():
    dqn_run(environment_type="gym")


if __name__ == "__main__":
    dqn_test()
# endregion
