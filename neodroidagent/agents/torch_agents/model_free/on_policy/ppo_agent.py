#!/usr/local/bin/python
# coding: utf-8
import copy
import math
import numpy
import torch
from torch.distributions import kl_divergence
from tqdm import tqdm

from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    DiscountedTransitionPoint,
    TransitionPoint,
    TransitionPointBuffer,
)
from neodroidagent.common.architectures.mlp_variants.actor_critic import ActorCriticMLP
from neodroidagent.utilities import (
    ActionSpaceNotSupported,
    discount_signal_torch,
    update_target,
)
from torch import nn

from draugr import MockWriter, Writer, batched_recycle, to_tensor, batch_generator_torch
from neodroidagent.utilities.misc.bool_tests import (
    is_none_or_zero_or_negative_or_mod_zero,
)
from neodroidagent.utilities.misc.checks import check_tensorised_shapes
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"

__all__ = ["PPOAgent"]


@super_init_pass_on_kws
class PPOAgent(TorchAgent):
    """
PPO, Proximal Policy Optimization method


"""

    # region Private

    def __init__(
        self,
        discount_factor=0.99,
        gae_tau=0.95,
        entropy_reg_coef=1e-3,
        value_reg_coef=5e-1,
        num_inner_updates=80,
        update_target_interval=1,
        clip_ratio=2e-1,
        copy_percentage=1.0,
        target_kl=1e-2,
        memory_buffer=TransitionPointBuffer(),
        critic_criterion: callable = nn.MSELoss(),
        optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=1e-3),
        arch_spec: GDKC = GDKC(ActorCriticMLP),
        **kwargs,
    ) -> None:
        """

:param discount_factor:
:param gae_tau:
:param actor_lr:
:param critic_lr:
:param entropy_reg_coef:
:param value_reg_coef:
:param num_inner_updates:
:param copy_percentage:
:param update_target_interval:
:param max_grad_norm:
:param solved_threshold:
:param test_interval:
:param early_stop:
:param rollouts:
:param clip_ratio:
:param state_type:
:param value_type:
:param action_type:
:param exploration_epsilon_start:
:param exploration_epsilon_end:
:param exploration_epsilon_decay:
:param kwargs:
"""
        super().__init__(**kwargs)

        self._copy_percentage = copy_percentage
        self._memory_buffer = memory_buffer
        self._optimiser_spec: GDKC = optimiser_spec
        self._arch_spec = arch_spec

        self._discount_factor = discount_factor
        self._gae_tau = gae_tau
        self._target_kl = target_kl

        self._entropy_reg_coef = entropy_reg_coef
        self._value_reg_coef = value_reg_coef
        self._num_inner_updates = num_inner_updates
        self._update_target_interval = update_target_interval
        self._critic_criterion = critic_criterion
        self._surrogate_clipping_value = clip_ratio

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
        """

    @param observation_space:
    @param action_space:
    @param signal_space:
    @param metric_writer:
    @param print_model_repr:
    @return:
    """

        self._arch_spec.kwargs["input_shape"] = self._input_shape
        if action_space.is_discrete:
            raise ActionSpaceNotSupported()

        self._arch_spec.kwargs["output_shape"] = self._output_shape

        # Construct actor and critic
        self.actor_critic = self._arch_spec().to(self._device)
        self._target_actor_critic = (
            copy.deepcopy(self.actor_critic).to(self._device).eval()
        )

        # Construct the optimizers for actor and critic
        self._optimiser = self._optimiser_spec(self.actor_critic.parameters())

    @property
    def models(self):
        """

    @return:
    """
        return {"actor_critic": self.actor_critic}

    # region Protected

    @drop_unused_kws
    def _sample(self, state) -> torch.tensor:
        """

    @param state:
    @return:
    """

        model_input = to_tensor(state, device=self._device, dtype=torch.float)
        with torch.no_grad():
            return self._target_actor_critic(model_input)[0].sample().detach()

    def extract_action(self, sample: torch.tensor) -> numpy.ndarray:
        """

    @param sample:
    @return:
    """
        return sample.to("cpu").numpy()

    @drop_unused_kws
    def _remember(self, *, signal, terminated, state, successor_state, sample):

        self._memory_buffer.add_transition_points(
            TransitionPoint(state, sample, successor_state, signal, terminated)
        )

    def _update_targets(
        self, update_percentage, *, metric_writer: Writer = None
    ) -> None:
        """

    @param update_percentage:
    @return:
    """
        if metric_writer:
            metric_writer.blip("Target Model Synced", self.update_i)

        update_target(
            target_model=self._target_actor_critic,
            source_model=self.actor_critic,
            copy_percentage=update_percentage,
        )

    @drop_unused_kws
    def _update(self, metric_writer: Writer = MockWriter()) -> float:
        """

    @param metric_writer:
    @return:
    """

        transitions = self._memory_buffer.sample_transition_points()
        self._memory_buffer.clear()

        with torch.no_grad():
            old_dist = self.actor_critic(
                to_tensor(transitions.state, device=self._device)
            )[0]
            action_log_prob = old_dist.log_prob(
                to_tensor(transitions.action, device=self._device)
            ).detach()

        discounted_signal = discount_signal_torch(
            to_tensor(transitions.signal, device=self._device),
            self._discount_factor,
            device=self.device,
            non_terminal=to_tensor(
                transitions.non_terminal_numerical, device=self._device
            ),
        )

        discounted_signal = (discounted_signal - discounted_signal.mean()) / (
            discounted_signal.std() + self._divide_by_zero_safety
        )

        discounted_tp = [
            DiscountedTransitionPoint(*a)
            for a in zip(
                transitions.state,
                transitions.action,
                transitions.successor_state,
                transitions.terminal,
                action_log_prob,
                discounted_signal,
            )
        ]

        accum_loss = 0

        loader = batched_recycle(discounted_tp)
        for i, mini_batch in zip(
            tqdm(range(self._num_inner_updates), desc="#Inner updates", leave=False),
            loader,
        ):
            loss, approx_kl = self.evaluate(
                DiscountedTransitionPoint(*zip(*mini_batch)),
                metric_writer=metric_writer,
            )

            self._optimiser.zero_grad()
            loss.backward()
            self.post_process_gradients(self.actor_critic)
            self._optimiser.step()

            if approx_kl > 1.5 * self._target_kl:
                break

            accum_loss += loss.detach().cpu().item()

        accum_loss /= self._num_inner_updates
        if metric_writer:
            metric_writer.scalar("loss", accum_loss)
            metric_writer.scalar("Inner Updates", i)

        if is_none_or_zero_or_negative_or_mod_zero(
            self._update_target_interval, self.update_i
        ):
            self._update_targets(self._copy_percentage, metric_writer=metric_writer)

        return accum_loss

    # endregion

    @drop_unused_kws
    def evaluate(
        self, batch: DiscountedTransitionPoint, *, metric_writer: Writer = None
    ) -> torch.tensor:
        """

    @param batch:
    @return:
    """
        tensorised = DiscountedTransitionPoint(
            *[to_tensor(a, device=self._device) for a in batch]
        )

        check_tensorised_shapes(tensorised)

        new_distribution, value_estimate = self.actor_critic(tensorised.state)
        action_log_probs_new = new_distribution.log_prob(tensorised.action)
        ratio = action_log_probs_new - tensorised.action_log_prob
        ratio = torch.clamp(ratio, -10, 10)
        ratio = ratio.exp()
        # if ratio explodes to (inf or Nan) due to the residual being to large check initialisation!
        # Generated action probs from (new policy) and (old policy).
        # Values of [0..1] means that actions less likely with the new policy,
        # while values [>1] mean action a more likely now
        clamped_ratio = torch.clamp(
            ratio,
            min=1.0 - self._surrogate_clipping_value,
            max=1.0 + self._surrogate_clipping_value,
        )

        advantage = tensorised.discounted_return - value_estimate.detach()
        policy_loss = -torch.min(ratio * advantage, clamped_ratio * advantage).mean()

        entropy_loss = -new_distribution.entropy().mean() * self._entropy_reg_coef

        value_error = (
            self._critic_criterion(value_estimate, tensorised.discounted_return)
            * self._value_reg_coef
        )

        approx_kl = (tensorised.action_log_prob - action_log_probs_new).mean().item()

        if metric_writer:
            metric_writer.scalar("advantage", advantage.mean().cpu().item())
            metric_writer.scalar("ratio", ratio.mean().detach().cpu().item())
            metric_writer.scalar("policy_loss", policy_loss.detach().cpu().item())
            metric_writer.scalar("entropy_loss", entropy_loss.detach().cpu().item())
            metric_writer.scalar("value_error", value_error.detach().cpu().item())
            metric_writer.scalar("approx_kl", approx_kl)

        return (value_error + policy_loss + entropy_loss), approx_kl
