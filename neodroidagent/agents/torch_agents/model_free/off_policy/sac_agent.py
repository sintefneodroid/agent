#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import copy
from typing import Any, Dict, Sequence, Tuple
from warnings import warn

import numpy
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from tqdm import tqdm

from draugr import MockWriter, Writer, to_tensor
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    Architecture,
    SamplePoint,
    ShallowStdNormalMLP,
    SingleHeadConcatInputMLP,
    TransitionPoint,
    TransitionPointBuffer,
)
from neodroidagent.utilities import is_zero_or_mod_zero, update_target
from neodroidagent.utilities.misc.checks import check_tensorised_shapes
from neodroidagent.utilities.misc.sampling import tanh_reparameterised_sample
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

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
        *,
        copy_percentage=3e-3,
        batch_size: int = 256,
        discount_factor: float = 0.99,
        target_update_interval: int = 1,
        num_inner_updates: int = 10,
        sac_alpha: float = 4e-2,
        memory_buffer=TransitionPointBuffer(1000000),
        auto_tune_sac_alpha=True,
        auto_tune_sac_alpha_optimiser_spec: GDKC = GDKC(
            constructor=torch.optim.Adam, lr=1e-3, eps=1e-4
        ),
        actor_optimiser_spec: GDKC = GDKC(
            constructor=torch.optim.Adam, lr=3e-4, eps=1e-4
        ),
        critic_optimiser_spec: GDKC = GDKC(
            constructor=torch.optim.Adam, lr=3e-4, eps=1e-4
        ),
        actor_arch_spec: GDKC = GDKC(ShallowStdNormalMLP),
        critic_arch_spec: GDKC = GDKC(SingleHeadConcatInputMLP),
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
    :param kwargs:
    """
        super().__init__(**kwargs)

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

    @drop_unused_kws
    def _remember(
        self, *, signal, terminated, state, successor_state, sample, **kwargs
    ):
        """

    @param signal:
    @param terminated:
    @param state:
    @param successor_state:
    @param sample:
    @param kwargs:
    @return:
    """

        self._memory_buffer.add_transition_points(
            TransitionPoint(state, sample.action, successor_state, signal, terminated)
        )

    @property
    def models(self) -> Dict[str, Architecture]:
        """

    @return:
    """
        return {
            "critic_1": self.critic_1,
            "critic_2": self.critic_2,
            "actor": self.actor,
        }

    def _sample(
        self,
        state: Any,
        *args,
        deterministic: bool = False,
        metric_writer: Writer = MockWriter(),
        **kwargs,
    ) -> Tuple[Sequence, Any]:
        """

    @param state:
    @param args:
    @param deterministic:
    @param metric_writer:
    @param kwargs:
    @return:
    """

        state = to_tensor(state, device=self._device)
        distribution = self.actor(state)

        with torch.no_grad():
            action = distribution.sample().detach()

        return SamplePoint(action, distribution)

    def extract_action(self, sample: SamplePoint) -> numpy.ndarray:
        """

    @param sample:
    @return:
    """
        return sample.action.to("cpu").numpy()

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Writer = MockWriter(),
        print_model_repr=True,
        *,
        critic_1=None,
        critic_1_optimizer=None,
        critic_2=None,
        critic_2_optimizer=None,
        actor=None,
        actor_optimizer=None,
    ) -> None:
        """

    @param observation_space:
    @param action_space:
    @param signal_space:
    @param metric_writer:
    @param print_model_repr:
    @param critic_1:
    @param critic_1_optimizer:
    @param critic_2:
    @param critic_2_optimizer:
    @param actor:
    @param actor_optimizer:
    @return:
    """
        # if action_space.is_discrete:
        #  raise ActionSpaceNotSupported()

        if not critic_1:
            self._critic_arch_spec.kwargs["input_shape"] = (
                *self._input_shape,
                *self._output_shape,
            )
            self._critic_arch_spec.kwargs["output_shape"] = 1
            self.critic_1 = self._critic_arch_spec().to(self._device)
            self.critic_1_optimiser = self._critic_optimiser_spec(
                self.critic_1.parameters()
            )
        else:
            self.critic_1 = critic_1
            self.critic_1_optimiser = critic_1_optimizer
        self.critic_1_target = copy.deepcopy(self.critic_1).to(self._device).eval()

        if not critic_2:
            self._critic_arch_spec.kwargs["input_shape"] = (
                *self._input_shape,
                *self._output_shape,
            )
            self._critic_arch_spec.kwargs["output_shape"] = 1
            self.critic_2 = self._critic_arch_spec().to(self._device)
            self.critic_2_optimiser = self._critic_optimiser_spec(
                self.critic_2.parameters()
            )
        else:
            self.critic_2 = critic_2
            self.critic_2_optimiser = critic_2_optimizer

        self.critic_2_target = copy.deepcopy(self.critic_2).to(self._device).eval()

        if not actor:
            self._actor_arch_spec.kwargs["input_shape"] = self._input_shape
            self._actor_arch_spec.kwargs["output_shape"] = self._output_shape
            self.actor = self._actor_arch_spec().to(self._device)
            self.actor_optimiser = self._actor_optimiser_spec(self.actor.parameters())
        else:
            self.actor = actor
            self.actor_optimiser = actor_optimizer

        if self._auto_tune_sac_alpha:
            self._target_entropy = -torch.prod(
                to_tensor(self._output_shape, device=self._device)
            ).item()
            self._log_sac_alpha = nn.Parameter(
                torch.log(to_tensor(self._sac_alpha, device=self._device)),
                requires_grad=True,
            )
            self.sac_alpha_optimiser = self._auto_tune_sac_alpha_optimiser_spec(
                [self._log_sac_alpha]
            )

    def on_load(self) -> None:
        """

    @return:
    """
        self.update_targets(1.0)

    def update_critics(
        self, tensorised: TransitionPoint, metric_writer: Writer = None
    ) -> float:
        """

    @param metric_writer:
    @param tensorised:
    @return:
    """
        with torch.no_grad():
            successor_action, log_prob = tanh_reparameterised_sample(
                self.actor(tensorised.successor_state)
            )
            min_successor_q = torch.min(
                self.critic_1_target(tensorised.successor_state, successor_action),
                self.critic_2_target(tensorised.successor_state, successor_action),
            )
            min_successor_q -= log_prob * self._sac_alpha

            successor_q_value = (
                tensorised.signal
                + tensorised.non_terminal_numerical
                * self._discount_factor
                * min_successor_q
            ).detach()

        q_value_loss1 = self._critic_criterion(
            self.critic_1(tensorised.state, tensorised.action), successor_q_value
        )
        self.critic_1_optimiser.zero_grad()
        q_value_loss1.backward()
        self.post_process_gradients(self.critic_1)
        self.critic_1_optimiser.step()

        q_value_loss2 = self._critic_criterion(
            self.critic_2(tensorised.state, tensorised.action), successor_q_value
        )
        self.critic_2_optimiser.zero_grad()
        q_value_loss2.backward()
        self.post_process_gradients(self.critic_2)
        self.critic_2_optimiser.step()

        out_loss = (q_value_loss1.detach() + q_value_loss2.detach()).cpu().item()

        if metric_writer:
            metric_writer.scalar("Critics_loss", out_loss)
            metric_writer.scalar("q_value_loss1", q_value_loss1.cpu().mean().item())
            metric_writer.scalar("q_value_loss2", q_value_loss2.cpu().mean().item())
            metric_writer.scalar("min_successor_q", min_successor_q.cpu().mean().item())
            metric_writer.scalar(
                "successor_q_value", successor_q_value.cpu().mean().item()
            )

        return out_loss

    def update_actor(self, tensorised: TransitionPoint, metric_writer=None) -> float:
        """

    @param tensorised:
    @param metric_writer:
    @return:
    """
        action, log_prob = tanh_reparameterised_sample(self.actor(tensorised.state))
        q_value = torch.min(
            self.critic_1(tensorised.state, action),
            self.critic_2(tensorised.state, action),
        )
        policy_loss = (log_prob * self._sac_alpha - q_value).mean()
        self.actor_optimiser.zero_grad()
        policy_loss.backward()
        self.post_process_gradients(self.actor)
        self.actor_optimiser.step()

        out_loss = policy_loss.detach().cpu().item()

        if metric_writer:
            metric_writer.scalar("Policy_loss", out_loss)
            metric_writer.scalar("log_prob", log_prob.cpu().mean().item())
            metric_writer.scalar("q_value", q_value.cpu().mean().item())

        return out_loss

    def update_alpha(
        self, tensorised: TransitionPoint, metric_writer: Writer = None
    ) -> float:
        """

    @param tensorised:
    @param metric_writer:
    @return:
    """
        _, log_prob = tanh_reparameterised_sample(self.actor(tensorised.state))
        alpha_loss = -(
            self._log_sac_alpha * (log_prob.detach() + self._target_entropy)
        ).mean()

        self.sac_alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.post_process_gradients(self._log_sac_alpha)
        self.sac_alpha_optimiser.step()

        self._sac_alpha = self._log_sac_alpha.exp()

        out_loss = alpha_loss.detach().cpu().item()

        if metric_writer:
            metric_writer.scalar("Sac_Alpha_Loss", out_loss)
            metric_writer.scalar("Sac_Alpha", self._sac_alpha.cpu().mean().item())

        return out_loss

    def _update(self, *args, metric_writer: Writer = MockWriter(), **kwargs) -> float:
        """

    @param args:
    @param metric_writer:
    @param kwargs:
    @return:
    """

        U = range(self._num_inner_updates)
        U = tqdm(U, desc="#Inner update", leave=False)
        accum_loss = 0
        for _ in U:
            tensorised = TransitionPoint(
                *[
                    to_tensor(a, device=self._device)
                    for a in self._memory_buffer.sample_transition_points(
                        self._batch_size
                    )
                ]
            )

            check_tensorised_shapes(tensorised)

            accum_loss += self.update_critics(tensorised, metric_writer=metric_writer)
            accum_loss += self.update_actor(tensorised, metric_writer=metric_writer)

            if self._auto_tune_sac_alpha:
                accum_loss += self.update_alpha(tensorised, metric_writer=metric_writer)

            if is_zero_or_mod_zero(self._target_update_interval, self.update_i):
                self.update_targets(self._copy_percentage, metric_writer=metric_writer)

        if metric_writer:
            metric_writer.scalar("Accum_loss", accum_loss)

        return accum_loss

    def update_targets(
        self, copy_percentage: float = 1e-2, *, metric_writer: Writer = None
    ) -> None:
        """

    Interpolation factor in polyak averaging for target networks. Target networks are updated towards main networks according to:

\theta_{\text{targ}} \leftarrow
\rho \theta_{\text{targ}} + (1-\rho) \theta

where \rho is polyak. (Always between 0 and 1, usually close to 1.)

    @param copy_percentage:
    @return:
    """
        if metric_writer:
            metric_writer.blip("Target Model Synced", self.update_i)

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
