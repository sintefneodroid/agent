#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Sequence, Tuple, Union

import torch
import torch.nn as nn

from draugr import global_torch_device, to_tensor
from draugr.writers import MockWriter
from draugr.writers.writer import Writer
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace
from neodroid.utilities.unity_specifications import EnvironmentSnapshot
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common.architectures import MLP
from neodroidagent.common.architectures.architecture import Architecture
from neodroidagent.common.architectures.experimental.merged import ConcatInputMLP
from neodroidagent.common.architectures.specialised.sac_architectures import (
    PolicyNetwork,
    QNetwork,
)
from neodroidagent.common.memory.replay_buffer import ReplayBufferNumpy
from neodroidagent.common.procedures.training.step_wise import StepWise
from neodroidagent.common.sessions import LinearSession
from neodroidagent.utilities import update_target
from neodroidagent.utilities.exploration import OrnsteinUhlenbeckProcess
from warg import drop_unused_kws
from warg.gdkc import GDKC
from warg.kw_passing import super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/5/19
           """
__all__ = ["SACAgent"]


@super_init_pass_on_kws
class SACAgent(TorchAgent):
    """
  Soft Actor Critic Agent

  https://arxiv.org/pdf/1801.01290.pdf
  https://arxiv.org/pdf/1812.05905.pdf
  """

    def __init__(
        self,
        copy_percentage=3e-3,
        signal_clipping=False,
        action_clipping=False,
        memory_buffer=ReplayBufferNumpy(1000000),
        actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        actor_arch_spec=GDKC(MLP),
        critic_arch_spec=GDKC(ConcatInputMLP),
        random_process_spec=GDKC(constructor=OrnsteinUhlenbeckProcess),
        **kwargs,
    ):
        """

:param copy_percentage:
:param signal_clipping:
:param action_clipping:
:param memory_buffer:
:param actor_optimiser_spec:
:param critic_optimiser_spec:
:param actor_arch_spec:
:param critic_arch_spec:
:param random_process_spec:
:param kwargs:
"""
        super().__init__(**kwargs)

        self._target_update_tau = copy_percentage
        self._signal_clipping = signal_clipping
        self._action_clipping = action_clipping
        self._memory = memory_buffer
        self._actor_optimiser_spec: GDKC = actor_optimiser_spec
        self._critic_optimiser_spec: GDKC = critic_optimiser_spec
        self._actor_arch_spec = actor_arch_spec
        self._critic_arch_spec = critic_arch_spec
        self._random_process_spec = random_process_spec

        self.q_criterion = nn.MSELoss()

    @drop_unused_kws
    def _remember(self, *, state, action, signal, next_state, terminal):
        self._memory.push(state, action, signal, next_state, terminal)

    @property
    def models(self) -> Dict[str, Architecture]:
        return {
            "critic_1": self.critic_1,
            "critic_2": self.critic_2,
            "policy_net": self.policy_net,
        }

    def _sample(
        self,
        state: Any,
        *args,
        no_random: bool = False,
        metric_writer: Writer = MockWriter(),
        **kwargs,
    ) -> Tuple[Sequence, Any]:
        with torch.no_grad():
            state = to_tensor(state, device=self._device)
            if no_random:
                *_, a = self.policy_net.sample(state)
            else:
                a, *_ = self.policy_net.sample(state)
            return a.detach().cpu().numpy()

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Writer = MockWriter(),
        print_model_repr=True,
    ) -> None:

        # if action_space.is_discrete:
        #  raise ActionSpaceNotSupported()

        action_dim = action_space.shape[0]
        state_dim = observation_space.shape[0]
        hidden_dim = 256

        self.critic_1 = QNetwork(state_dim, action_dim, hidden_dim).to(self._device)
        self.critic_1_target = QNetwork(state_dim, action_dim, hidden_dim).to(
            self._device
        )

        self.critic_2 = QNetwork(state_dim, action_dim, hidden_dim).to(self._device)
        self.critic_2_target = QNetwork(state_dim, action_dim, hidden_dim).to(
            self._device
        )

        self.policy_net = PolicyNetwork(
            state_dim,
            action_dim,
            hidden_dim,
            # action_space=action_space
        ).to(self._device)

        self.soft_q_optimizer1 = self._critic_optimiser_spec(self.critic_1.parameters())
        self.soft_q_optimizer2 = self._critic_optimiser_spec(self.critic_2.parameters())
        self.policy_optimizer = self._actor_optimiser_spec(self.policy_net.parameters())

        self.update_targets(1.0)

    def on_load(self):
        self.update_targets(1.0)

    def _update(
        self,
        *args,
        batch_size: int = 128,
        discount_factor: float = 0.99,
        target_update_interval: int = 1,
        alpha=0,
        metric_writer: Writer = MockWriter(),
        **kwargs,
    ) -> None:
        state, action, signal, next_state, terminals = self._memory.sample(batch_size)

        state = to_tensor(state, device=self._device)
        next_state = to_tensor(next_state, device=self._device)
        action = to_tensor(action, device=self._device)
        signal = to_tensor(signal, device=self._device).unsqueeze(1)
        terminals = to_tensor(terminals, torch.float, device=self._device).unsqueeze(1)

        with torch.no_grad():
            new_action, next_log_prob, *_ = self.policy_net.sample(next_state)
            min_q_next_value_target = (
                torch.min(
                    self.critic_1_target(next_state, new_action),
                    self.critic_2_target(next_state, new_action),
                )
                - alpha * next_log_prob
            )
            next_q_value = (
                signal + (1 - terminals) * discount_factor * min_q_next_value_target
            )

        self.soft_q_optimizer1.zero_grad()
        q_value_loss1 = self.q_criterion(self.critic_1(state, action), next_q_value)
        q_value_loss1.backward()
        self.soft_q_optimizer1.step()

        self.soft_q_optimizer2.zero_grad()
        q_value_loss2 = self.q_criterion(self.critic_2(state, action), next_q_value)
        q_value_loss2.backward()
        self.soft_q_optimizer2.step()

        self.policy_optimizer.zero_grad()
        naction, naction_log_prob, *_ = self.policy_net.sample(state)
        q_value = torch.min(
            self.critic_1(state, naction), self.critic_2(state, naction)
        )
        policy_loss = ((alpha * naction_log_prob) - q_value).mean()
        policy_loss.backward()
        self.policy_optimizer.step()

        if False:  # self.automatic_entropy_tuning
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        if self._update_i % target_update_interval == 0:
            self.update_targets(self._target_update_tau)

    def update_targets(self, soft_update_percentage: float = 1e-2):
        update_target(
            target_model=self.critic_1_target,
            source_model=self.critic_1,
            copy_percentage=soft_update_percentage,
        )

        update_target(
            target_model=self.critic_2_target,
            source_model=self.critic_2,
            copy_percentage=soft_update_percentage,
        )


# region Test
def sac_test():
    sac_run(environment_type="gym")


def sac_run(
    rollouts=None, skip: bool = True, environment_type: Union[bool, str] = True
):
    from neodroidagent.common.sessions import session_entry_point
    from . import sac_test_config as C

    if rollouts:
        C.ROLLOUTS = rollouts

    session_entry_point(
        SACAgent,
        C,
        session=LinearSession(
            procedure=StepWise,
            environment_name=C.ENVIRONMENT_NAME,
            auto_reset_on_terminal_state=True,
            environment_type=environment_type,
        ),
        skip_confirmation=skip,
    )


if __name__ == "__main__":
    sac_test()

# endif
