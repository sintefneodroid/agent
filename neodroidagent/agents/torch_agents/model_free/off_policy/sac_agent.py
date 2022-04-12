#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
import itertools
from typing import Any, Dict, Optional, Tuple

import numpy
import torch
from draugr.torch_utilities import (
    freeze_model,
    frozen_parameters,
    to_scalar,
    to_tensor,
    ShallowStdNormalMLP,
    PreConcatInputMLP,
    Architecture,
)
from draugr.writers import MockWriter, Writer
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    Memory,
    SamplePoint,
    TransitionPoint,
    TransitionPointBuffer,
)
from neodroidagent.utilities import (
    ActionSpaceNotSupported,
    update_target,
)
from neodroidagent.utilities.misc.sampling import normal_tanh_reparameterised_sample
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Optimizer
from tqdm import tqdm
from trolls.spaces import ActionSpace, ObservationSpace, SignalSpace
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws, is_zero_or_mod_zero

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/5/19
           """
__all__ = ["SoftActorCriticAgent"]


@super_init_pass_on_kws
class SoftActorCriticAgent(TorchAgent):
    """
    Soft Actor Critic Agent

    https://arxiv.org/pdf/1801.01290.pdf
    https://arxiv.org/pdf/1812.05905.pdf"""

    def __init__(
        self,
        *,
        copy_percentage: float = 1e-2,
        batch_size: int = 100,
        discount_factor: float = 0.999,
        target_update_interval: int = 1,
        num_inner_updates: int = 20,
        sac_alpha: float = 1e-2,
        memory_buffer: Memory = TransitionPointBuffer(1000000),
        auto_tune_sac_alpha: bool = False,
        auto_tune_sac_alpha_optimiser_spec: GDKC = GDKC(
            constructor=torch.optim.Adam, lr=3e-4
        ),
        actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        actor_arch_spec: GDKC = GDKC(
            ShallowStdNormalMLP, mean_head_activation=torch.tanh
        ),
        critic_arch_spec: GDKC = GDKC(PreConcatInputMLP),
        critic_criterion: callable = mse_loss,
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
        :param kwargs:"""
        super().__init__(**kwargs)

        assert 0 <= discount_factor <= 1.0
        assert 0 <= copy_percentage <= 1.0

        self._batch_size = batch_size
        self._discount_factor = discount_factor
        self._target_update_interval = target_update_interval
        self._sac_alpha = sac_alpha
        self._copy_percentage = copy_percentage
        self._memory_buffer = memory_buffer
        self._actor_optimiser_spec: GDKC = actor_optimiser_spec
        self._critic_optimiser_spec: GDKC = critic_optimiser_spec
        self._actor_arch_spec = actor_arch_spec
        self._critic_arch_spec = critic_arch_spec

        self._num_inner_updates = num_inner_updates
        self._critic_criterion = critic_criterion

        self._auto_tune_sac_alpha = auto_tune_sac_alpha
        self._auto_tune_sac_alpha_optimiser_spec = auto_tune_sac_alpha_optimiser_spec
        self.inner_update_i = 0

    @drop_unused_kws
    def _remember(
        self,
        *,
        signal: Any,
        terminated: Any,
        state: Any,
        successor_state: Any,
        sample: Any,
    ) -> None:
        """

        :param signal:
        :param terminated:
        :param state:
        :param successor_state:
        :param sample:
        :param kwargs:
        :return:"""
        a = [
            TransitionPoint(*s)
            for s in zip(state, sample[0], successor_state, signal, terminated)
        ]
        for a_ in a:
            self._memory_buffer.add_transition_point(a_)

    @property
    def models(self) -> Dict[str, Architecture]:
        """

        :return:"""
        return {
            "critic_1": self.critic_1,
            "critic_2": self.critic_2,
            "actor": self.actor,
        }

    @property
    def optimisers(self) -> Dict[str, Optimizer]:
        return {
            "actor_optimiser": self.actor_optimiser,
            "critic_optimiser": self.critic_optimiser,
        }

    @drop_unused_kws
    def _sample(
        self,
        state: Any,
        *args,
        deterministic: bool = False,
        metric_writer: Optional[Writer] = MockWriter(),
    ) -> Tuple[torch.Tensor, Any]:
        """

        :param state:
        :param args:
        :param deterministic:
        :param metric_writer:
        :param kwargs:
        :return:"""
        distribution = self.actor(to_tensor(state, device=self._device, dtype=float))

        with torch.no_grad():
            return (torch.tanh(distribution.sample().detach()), distribution)

    def extract_action(self, sample: SamplePoint) -> numpy.ndarray:
        """

        :param sample:
        :return:"""
        return sample[0].to("cpu").numpy()

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Optional[Writer] = MockWriter(),
        print_model_repr: bool = True,
    ) -> None:
        """

        :param observation_space:
        :param action_space:
        :param signal_space:
        :param metric_writer:
        :param print_model_repr:
        :return:"""
        if action_space.is_singular_discrete:
            raise ActionSpaceNotSupported(
                "discrete action space not supported in this implementation"
            )

        self._critic_arch_spec.kwargs["input_shape"] = (
            self._input_shape + self._output_shape
        )
        self._critic_arch_spec.kwargs["output_shape"] = 1

        self.critic_1 = self._critic_arch_spec().to(self._device)
        self.critic_1_target = copy.deepcopy(self.critic_1).to(self._device)
        freeze_model(self.critic_1_target, True, True)

        self.critic_2 = self._critic_arch_spec().to(self._device)
        self.critic_2_target = copy.deepcopy(self.critic_2).to(self._device)
        freeze_model(self.critic_2_target, True, True)

        self.critic_optimiser = self._critic_optimiser_spec(
            itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())
        )

        self._actor_arch_spec.kwargs["input_shape"] = self._input_shape
        self._actor_arch_spec.kwargs["output_shape"] = self._output_shape
        self.actor = self._actor_arch_spec().to(self._device)
        self.actor_optimiser = self._actor_optimiser_spec(self.actor.parameters())

        if self._auto_tune_sac_alpha:
            self._target_entropy = -torch.prod(
                to_tensor(self._output_shape, device=self._device, dtype=float)
            ).item()
            self._log_sac_alpha = nn.Parameter(
                torch.log(to_tensor(self._sac_alpha, device=self._device, dtype=float)),
                requires_grad=True,
            )
            self.sac_alpha_optimiser = self._auto_tune_sac_alpha_optimiser_spec(
                [self._log_sac_alpha]
            )

    def on_load(self) -> None:
        """

        :return:"""
        self.update_targets(1.0)

    def update_critics(
        self, tensorised: TransitionPoint, metric_writer: Optional[Writer] = None
    ) -> float:
        """

        :param metric_writer:
        :param tensorised:
        :return:"""
        with torch.no_grad():
            successor_action, successor_log_prob = normal_tanh_reparameterised_sample(
                self.actor(tensorised.successor_state)
            )

            min_successor_q = (
                torch.min(
                    self.critic_1_target(tensorised.successor_state, successor_action),
                    self.critic_2_target(tensorised.successor_state, successor_action),
                )
                - successor_log_prob * self._sac_alpha
            )

            successor_q_value = (
                tensorised.signal
                + tensorised.non_terminal_numerical
                * self._discount_factor
                * min_successor_q
            ).detach()
            assert not successor_q_value.requires_grad

        q_value_loss1 = self._critic_criterion(
            self.critic_1(tensorised.state, tensorised.action), successor_q_value
        )
        q_value_loss2 = self._critic_criterion(
            self.critic_2(tensorised.state, tensorised.action), successor_q_value
        )
        critic_loss = q_value_loss1 + q_value_loss2
        assert critic_loss.requires_grad
        self.critic_optimiser.zero_grad()
        critic_loss.backward()
        self.post_process_gradients(self.critic_1.parameters())
        self.post_process_gradients(self.critic_2.parameters())
        self.critic_optimiser.step()

        out_loss = to_scalar(critic_loss)

        if metric_writer:
            metric_writer.scalar("Critics_loss", out_loss, self.update_i)
            metric_writer.scalar(
                "q_value_loss1", to_scalar(q_value_loss1), self.update_i
            )
            metric_writer.scalar(
                "q_value_loss2", to_scalar(q_value_loss2), self.update_i
            )
            metric_writer.scalar(
                "min_successor_q", to_scalar(min_successor_q), self.update_i
            )
            metric_writer.scalar(
                "successor_q_value", to_scalar(successor_q_value), self.update_i
            )

        return out_loss

    def update_actor(
        self, tensorised: torch.Tensor, metric_writer: Optional[Writer] = None
    ) -> float:
        """

        :param tensorised:
        :param metric_writer:
        :return:"""

        dist = self.actor(tensorised.state)
        action, log_prob = normal_tanh_reparameterised_sample(dist)

        # Check gradient paths
        assert action.requires_grad
        assert log_prob.requires_grad

        q_values = (
            self.critic_1(tensorised.state, action),
            self.critic_2(tensorised.state, action),
        )
        assert q_values[0].requires_grad and q_values[1].requires_grad

        policy_loss = torch.mean(self._sac_alpha * log_prob - torch.min(*q_values))
        self.actor_optimiser.zero_grad()
        policy_loss.backward()
        self.post_process_gradients(self.actor.parameters())
        self.actor_optimiser.step()

        out_loss = to_scalar(policy_loss)

        if metric_writer:
            metric_writer.scalar("Policy_loss", out_loss)
            metric_writer.scalar("q_value_1", to_scalar(q_values[0]))
            metric_writer.scalar("q_value_2", to_scalar(q_values[1]))
            metric_writer.scalar("policy_stddev", to_scalar(dist.stddev))
            metric_writer.scalar("policy_log_prob", to_scalar(log_prob))

        if self._auto_tune_sac_alpha:
            out_loss += self.update_alpha(
                log_prob.detach(), metric_writer=metric_writer
            )

        return out_loss

    def update_alpha(
        self, log_prob: torch.Tensor, metric_writer: Optional[Writer] = None
    ) -> float:
        """

        :param log_prob:
        :type log_prob:
        :param tensorised:
        :param metric_writer:
        :return:"""
        assert not log_prob.requires_grad

        alpha_loss = -torch.mean(
            self._log_sac_alpha * (log_prob + self._target_entropy)
        )

        self.sac_alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.post_process_gradients(self._log_sac_alpha)
        self.sac_alpha_optimiser.step()

        self._sac_alpha = self._log_sac_alpha.exp()

        out_loss = alpha_loss.detach().cpu().item()

        if metric_writer:
            metric_writer.scalar("Sac_Alpha_Loss", out_loss, self.update_i)
            metric_writer.scalar("Sac_Alpha", to_scalar(self._sac_alpha), self.update_i)

        return out_loss

    def _update(
        self, *args, metric_writer: Optional[Writer] = MockWriter(), **kwargs
    ) -> float:
        """

        :param args:
        :param metric_writer:
        :param kwargs:
        :return:"""
        accum_loss = 0
        for ith_inner_update in tqdm(
            range(self._num_inner_updates),
            desc="Inner update #",
            leave=False,
            postfix=f"Agent update #{self.update_i}",
        ):
            self.inner_update_i += 1
            batch = self._memory_buffer.sample(self._batch_size)
            tensorised = TransitionPoint(
                *[to_tensor(a, device=self._device, dtype=float) for a in batch]
            )

            with frozen_parameters(self.actor.parameters()):
                accum_loss += self.update_critics(
                    tensorised, metric_writer=metric_writer
                )

            with frozen_parameters(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters())
            ):
                accum_loss += self.update_actor(tensorised, metric_writer=metric_writer)

            if is_zero_or_mod_zero(self._target_update_interval, self.inner_update_i):
                self.update_targets(self._copy_percentage, metric_writer=metric_writer)

        if metric_writer:
            metric_writer.scalar("Accum_loss", accum_loss, self.update_i)
            metric_writer.scalar("num_inner_updates_i", ith_inner_update, self.update_i)

        return accum_loss

    def update_targets(
        self, copy_percentage: float = 0.005, *, metric_writer: Optional[Writer] = None
    ) -> None:
        """

        Interpolation factor in polyak averaging for target networks. Target networks are updated towards main
        networks according to:

        \theta_{\text{targ}} \leftarrow
        \rho \theta_{\text{targ}} + (1-\rho) \theta

        where \rho is polyak. (Always between 0 and 1, usually close to 1.)

        :param metric_writer:
        :type metric_writer:
        :param copy_percentage:
        :return:"""
        if metric_writer:
            metric_writer.blip("Target Models Synced", self.update_i)

        update_target(
            target_model=self.critic_1_target,
            source_model=self.critic_1,
            copy_percentage=copy_percentage,
        )

        update_target(
            target_model=self.critic_2_target,
            source_model=self.critic_2,
            copy_percentage=copy_percentage,
        )
