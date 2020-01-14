#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Any, Sequence, Union

import numpy
import torch
import torch.nn.functional as F
from numpy import mean
from tqdm import tqdm

from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter
from draugr.writers.writer import Writer
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common.architectures.experimental.merged import (
    SingleHeadMergedInputMLP,
)
from neodroidagent.common.architectures.mlp import SingleHeadMLP
from neodroidagent.common.memory import TransitionBuffer
from neodroidagent.common.procedures.training import Batched
from neodroidagent.utilities import update_target
from neodroidagent.utilities.exceptions.exceptions import ActionSpaceNotSupported
from neodroidagent.utilities.exploration.sampling import OrnsteinUhlenbeckProcess
from warg import drop_unused_kws
from warg.gdkc import GDKC
from warg.kw_passing import super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__all__ = ["DDPGAgent"]

tqdm.monitor_interval = 0


@super_init_pass_on_kws
class DDPGAgent(TorchAgent):
    """
The Deep Deterministic Policy Gradient (DDPG) Agent

Parameters
----------
  actor_optimizer_spec: OptimiserSpec
      Specifying the constructor and kwargs, as well as learning rate and other
      parameters for the optimiser
  critic_optimizer_spec: OptimiserSpec
  num_feature: int
      The number of features of the environmental state
  num_action: int
      The number of available actions that agent can choose from
  replay_memory_size: int
      How many memories to store in the replay memory.
  batch_size: int
      How many transitions to sample each time experience is replayed.
  tau: float
      The update rate that target networks slowly track the learned networks.
"""

    @drop_unused_kws
    def _remember(self, *, transitions):
        self._memory_buffer.add_transitions(transitions)

    @property
    def models(self):
        return {"_actor": self._actor, "_critic": self._critic}

    # region Private

    def __init__(
        self,
        random_process_spec=GDKC(constructor=OrnsteinUhlenbeckProcess),
        memory_buffer=TransitionBuffer(1000000),
        evaluation_function=F.smooth_l1_loss,
        actor_arch_spec=GDKC(SingleHeadMLP),
        critic_arch_spec=GDKC(SingleHeadMergedInputMLP),
        discount_factor=0.95,
        sync_target_model_frequency=10000,
        state_type=torch.float,
        value_type=torch.float,
        action_type=torch.float,
        exploration_epsilon_start=0.9,
        exploration_epsilon_end=0.05,
        exploration_epsilon_decay=10000,
        early_stopping_condition=None,
        batch_size=64,
        noise_factor=3e-1,
        low_action_clip=-1.0,
        high_action_clip=1.0,
        copy_percentage=3e-3,
        signal_clipping=False,
        action_clipping=False,
        actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        **kwargs
    ):
        """

:param random_process_spec:
:param memory_buffer:
:param evaluation_function:
:param actor_arch_spec:
:param critic_arch_spec:
:param discount_factor:
:param initial_observation_period:
:param learning_frequency:
:param sync_target_model_frequency:
:param state_type:
:param value_type:
:param action_type:
:param exploration_epsilon_start:
:param exploration_epsilon_end:
:param exploration_epsilon_decay:
:param early_stopping_condition:
:param batch_size:
:param noise_factor:
:param low_action_clip:
:param high_action_clip:
:param kwargs:
"""
        super().__init__(**kwargs)

        self._target_update_tau = copy_percentage
        self._signal_clipping = signal_clipping
        self._action_clipping = action_clipping
        self._actor_optimiser_spec: GDKC = actor_optimiser_spec
        self._critic_optimiser_spec: GDKC = critic_optimiser_spec
        self._actor_arch_spec = actor_arch_spec
        self._critic_arch_spec = critic_arch_spec
        self._random_process_spec = random_process_spec

        # Adds noise for exploration

        # self._memory = U.PrioritisedReplayMemory(config.REPLAY_MEMORY_SIZE)  # Cuda trouble
        self._memory_buffer = memory_buffer
        self._evaluation_function = evaluation_function
        self._actor_arch_spec = actor_arch_spec
        self._critic_arch_spec = critic_arch_spec
        self._discount_factor = discount_factor
        self._sync_target_model_frequency = sync_target_model_frequency
        self._state_type = state_type
        self._value_type = value_type
        self._action_type = action_type
        self._exploration_epsilon_start = exploration_epsilon_start
        self._exploration_epsilon_end = exploration_epsilon_end
        self._exploration_epsilon_decay = exploration_epsilon_decay
        self._early_stopping_condition = early_stopping_condition
        self._batch_size = batch_size
        self._noise_factor = noise_factor
        self._low_action_clip = low_action_clip
        self._high_action_clip = high_action_clip

        (
            self._actor,
            self._target_actor,
            self._critic,
            self._target_critic,
            self._actor_optimiser,
            self._critic_optimiser,
        ) = (None, None, None, None, None, None)

        self._random_process = None

    # endregion

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Writer = MockWriter(),
        print_model_repr=True,
    ) -> None:

        self._actor_arch_spec.kwargs["input_shape"] = self._input_shape
        if action_space.is_discrete:
            raise ActionSpaceNotSupported()

        self._actor_arch_spec.kwargs["output_shape"] = self._output_shape

        self._critic_arch_spec.kwargs["input_shape"] = (
            *self._input_shape,
            *self._output_shape,
        )
        # self._actor_arch_spec = GDCS(MergedInputMLP, self._critic_arch_spec.kwargs)
        self._critic_arch_spec.kwargs["output_shape"] = 1

        # Construct actor and critic
        self._actor = self._actor_arch_spec().to(self._device)
        self._target_actor = self._actor_arch_spec().to(self._device).eval()

        self._critic = self._critic_arch_spec().to(self._device)
        self._target_critic = self._critic_arch_spec().to(self._device).eval()

        self._random_process = self._random_process_spec(
            sigma=mean([r.span for r in action_space.ranges])
        )

        # Construct the optimizers for actor and critic
        self._actor_optimiser = self._actor_optimiser_spec(self._actor.parameters())
        self._critic_optimiser = self._critic_optimiser_spec(self._critic.parameters())

    # region Public

    def evaluate(self, batch):
        """

:param batch:
:type kwargs: object
"""
        (
            state_batch,
            action_batch,
            signal_batch,
            next_state_batch,
            non_terminal_batch,
        ) = batch
        states = to_tensor(state_batch, device=self._device, dtype=self._state_type)
        next_states = to_tensor(
            next_state_batch, device=self._device, dtype=self._state_type
        )
        actions = to_tensor(action_batch, device=self._device, dtype=self._action_type)
        signals = to_tensor(signal_batch, device=self._device, dtype=self._value_type)
        non_terminal_mask = to_tensor(
            non_terminal_batch, device=self._device, dtype=self._value_type
        )

        ### Critic ###
        # Compute current Q value, critic takes state and action chosen
        Q_current = self._critic(states, actions)
        # Compute next Q value based on which action target actor would choose
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        with torch.no_grad():
            target_actions = self._target_actor(states)
            next_max_q = self._target_critic(next_states, target_actions)

        next_Q_values = non_terminal_mask * next_max_q.view(next_max_q.shape[0], -1)

        Q_target = signals + (
            self._discount_factor * next_Q_values
        )  # Compute the target of the current Q values

        td_error = self._evaluation_function(
            Q_current.view(Q_current.shape[0], -1), Q_target
        )  # Compute Bellman error (using Huber loss)

        return td_error, states

    def update_targets(self):
        update_target(
            target_model=self._target_critic,
            source_model=self._critic,
            copy_percentage=self._target_update_tau,
        )
        update_target(
            target_model=self._target_actor,
            source_model=self._actor,
            copy_percentage=self._target_update_tau,
        )

    # endregion

    # region Protected

    @drop_unused_kws
    def _update(self, *, metric_writer: Writer = MockWriter()):
        """
Update the target networks

:return:
:rtype:
"""
        if len(self._memory_buffer) < self._batch_size:
            return

        self._critic_optimiser.zero_grad()

        batch = self._memory_buffer.sample_transitions(self._batch_size)
        td_error, state_batch_var = self.evaluate(batch)

        td_error.backward()
        self._critic_optimiser.step()  # Optimize the critic

        self._actor_optimiser.zero_grad()
        action_batch = self._actor(state_batch_var)
        c = self._critic(state_batch_var, action_batch)
        loss = -c.mean()

        loss.backward()
        self._actor_optimiser.step()  # Optimize the actor

        self.update_targets()

        if metric_writer:
            metric_writer.scalar("td_error", td_error.cpu().item())
            metric_writer.scalar("critic_loss", loss)

        self._memory_buffer.clear()

        return td_error, loss

    @drop_unused_kws
    def _sample(self, state: Sequence) -> Any:
        # self._random_process.reset()

        state = to_tensor(state, device=self._device, dtype=self._state_type)

        with torch.no_grad():
            action_out = self._actor(state).detach().to("cpu").numpy()

        # Add action space noise for exploration, alternative is parameter space noise
        noise = self._random_process.sample(action_out.shape)
        action_out += noise * self._noise_factor

        if self._action_clipping:
            action_out = numpy.clip(
                action_out, self._low_action_clip, self._high_action_clip
            )

        return action_out, None, None

    # endregion


# region Test


def ddpg_run(
    rollouts=None, skip: bool = True, environment_type: Union[bool, str] = True
):
    from neodroidagent.common.sessions.session_entry_point import session_entry_point
    from neodroidagent.common.sessions.single_agent.parallel import ParallelSession
    from . import ddpg_test_config as C

    if rollouts:
        C.ROLLOUTS = rollouts

    session_entry_point(
        DDPGAgent,
        C,
        session=ParallelSession(
            procedure=Batched,
            environment_name=C.ENVIRONMENT_NAME,
            environment_type=environment_type,
            auto_reset_on_terminal_state=True,
        ),
        skip_confirmation=skip,
        environment_type=environment_type,
    )


def ddpg_test():
    ddpg_run(environment_type="gym")


if __name__ == "__main__":
    ddpg_test()

    # ddpg_run()
# endregion
