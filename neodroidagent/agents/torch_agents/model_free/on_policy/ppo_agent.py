#!/usr/local/bin/python
# coding: utf-8
import copy
from typing import Any, Tuple

import numpy
import torch
from torch.nn.functional import mse_loss
from tqdm import tqdm

from draugr import MockWriter, Writer, freeze_model, shuffled_batches, to_tensor
from draugr.metrics.accumulation import mean_accumulator
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.agent import ClipFeature
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    ActorCriticMLP,
    CategoricalActorCriticMLP,
    TransitionPointTrajectoryBuffer,
    ValuedTransitionPoint,
)
from neodroidagent.utilities import (
    ActionSpaceNotSupported,
    Distribution,
    is_none_or_zero_or_negative_or_mod_zero,
    torch_compute_gae,
    update_target,
)
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
"""
__all__ = ["PPOAgent"]


@super_init_pass_on_kws
class PPOAgent(TorchAgent):
    r"""
PPO, Proximal Policy Optimization method

https://arxiv.org/abs/1707.06347 - PPO
https://arxiv.org/abs/1506.02438 - Advantage

https://spinningup.openai.com/en/latest/algorithms/ppo.html

"""

    def __init__(
        self,
        discount_factor: float = 0.95,
        gae_lambda: float = 0.95,
        entropy_reg_coef: float = 0,
        value_reg_coef: float = 5e-1,
        num_inner_updates: int = 10,
        mini_batch_size: int = 64,
        update_target_interval: int = 1,
        clip_ratio: float = 2e-1,
        copy_percentage: float = 1.0,
        target_kl: float = 1e-2,
        memory_buffer: Any = TransitionPointTrajectoryBuffer(),
        critic_criterion: callable = mse_loss,
        optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        continuous_arch_spec: GDKC = GDKC(ActorCriticMLP),
        discrete_arch_spec: GDKC = GDKC(CategoricalActorCriticMLP),
        gradient_norm_clipping: ClipFeature = ClipFeature(True, -0.5, 0.5),
        **kwargs,
    ) -> None:
        """

:param discount_factor:
:param gae_lambda:
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
        super().__init__(gradient_norm_clipping=gradient_norm_clipping, **kwargs)

        assert 0 <= discount_factor <= 1.0
        assert 0 <= gae_lambda <= 1.0

        self._copy_percentage = copy_percentage
        self._memory_buffer = memory_buffer
        self._optimiser_spec: GDKC = optimiser_spec
        self._continuous_arch_spec = continuous_arch_spec
        self._discrete_arch_spec = discrete_arch_spec

        self._discount_factor = discount_factor
        self._gae_lambda = gae_lambda
        self._target_kl = target_kl

        self._mini_batch_size = mini_batch_size
        self._entropy_reg_coef = entropy_reg_coef
        self._value_reg_coef = value_reg_coef
        self._num_inner_updates = num_inner_updates
        self._update_target_interval = update_target_interval
        self._critic_criterion = critic_criterion
        self._surrogate_clipping_value = clip_ratio
        self.inner_update_i = 0

    @drop_unused_kws
    def __build__(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        metric_writer: Writer = MockWriter(),
        print_model_repr: bool = True,
    ) -> None:
        """

@param observation_space:
@param action_space:
@param signal_space:
@param metric_writer:
@param print_model_repr:
@return:
"""
        if action_space.is_mixed:
            raise ActionSpaceNotSupported()
        elif action_space.is_continuous:
            self._continuous_arch_spec.kwargs["input_shape"] = self._input_shape
            self._continuous_arch_spec.kwargs["output_shape"] = self._output_shape
            self.actor_critic = self._continuous_arch_spec().to(self._device)
        else:
            self._discrete_arch_spec.kwargs["input_shape"] = self._input_shape
            self._discrete_arch_spec.kwargs["output_shape"] = self._output_shape
            self.actor_critic = self._discrete_arch_spec().to(self._device)

        self._target_actor_critic = copy.deepcopy(self.actor_critic).to(self._device)
        freeze_model(self._target_actor_critic, True, True)

        self._optimiser = self._optimiser_spec(self.actor_critic.parameters())

    @property
    def models(self) -> dict:
        """

@return:
"""
        return {"actor_critic": self.actor_critic}

    # region Protected

    @drop_unused_kws
    def _sample(self, state: numpy.ndarray, deterministic: bool = False) -> Tuple:
        """

@param state:
@return:
"""
        with torch.no_grad():
            dist, val_est = self._target_actor_critic(
                to_tensor(state, device=self._device, dtype=torch.float)
            )

            if deterministic:
                if self.action_space.is_discrete:
                    action = dist.logits.max(-1)[-1]
                else:
                    action = dist.mean
            else:
                action = dist.sample()

            if self.action_space.is_discrete:
                action = action.unsqueeze(-1)
            else:
                action = torch.clamp(action, -1, 1)

        return action.detach(), dist, val_est.detach()

    def extract_action(self, sample: torch.tensor) -> numpy.ndarray:
        """

@param sample:
@return:
"""
        return sample[0].to("cpu").numpy()

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
        self._memory_buffer.add_transition_point(
            ValuedTransitionPoint(
                state,
                sample[0],
                successor_state,
                signal,
                terminated,
                sample[1],
                sample[2],
            )
        )

    def _update_targets(
        self, copy_percentage: float, *, metric_writer: Writer = None
    ) -> None:
        """

@param copy_percentage:
@return:
"""
        if metric_writer:
            metric_writer.blip("Target Model Synced", self.update_i)

        update_target(
            target_model=self._target_actor_critic,
            source_model=self.actor_critic,
            copy_percentage=copy_percentage,
        )

    def get_log_prob(self, dist: Distribution, action: torch.tensor) -> torch.tensor:
        if self.action_space.is_discrete:
            return dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        else:
            return dist.log_prob(action).sum(axis=-1, keepdims=True)

    def _prepare_transitions(self):
        transitions = self._memory_buffer.sample()
        self._memory_buffer.clear()

        signal = to_tensor(transitions.signal, device=self._device)
        non_terminal = to_tensor(
            transitions.non_terminal_numerical, device=self._device
        )
        state = to_tensor(transitions.state, device=self.device)
        action = to_tensor(transitions.action, device=self.device)
        value_estimate_target = to_tensor(
            transitions.value_estimate, device=self._device
        )

        action_log_prob_old = to_tensor(
            [
                self.get_log_prob(dist, a)
                for dist, a in zip(transitions.distribution, transitions.action)
            ],
            device=self.device,
        )

        with torch.no_grad():
            *_, successor_value_estimate = self.actor_critic(
                to_tensor((transitions.successor_state[-1],), device=self.device)
            )
            value_estimate_target = torch.cat(
                (value_estimate_target, successor_value_estimate), dim=0
            )

            discounted_signal, advantage = torch_compute_gae(
                signal,
                non_terminal,
                value_estimate_target,
                discount_factor=self._discount_factor,
                gae_lambda=self._gae_lambda,
                device=self.device,
            )

        return (
            state.flatten(0, 1),
            action.flatten(0, 1),
            action_log_prob_old.flatten(0, 1),
            discounted_signal.flatten(0, 1),
            advantage.flatten(0, 1),
        )

    @drop_unused_kws
    def _update(self, metric_writer: Writer = MockWriter()) -> float:
        """

@param metric_writer:
@return:
"""
        transitions = self._prepare_transitions()

        accum_loss = mean_accumulator()
        for i in tqdm(
            range(self._num_inner_updates), desc="#Inner updates", leave=False
        ):
            self.inner_update_i += 1
            loss, early_stop_inner = self.inner_update(
                *transitions, metric_writer=metric_writer
            )
            accum_loss.send(loss)

            if is_none_or_zero_or_negative_or_mod_zero(
                self._update_target_interval, self.inner_update_i
            ):
                self._update_targets(self._copy_percentage, metric_writer=metric_writer)

            if early_stop_inner:
                break

        mean_loss = accum_loss.send()

        if metric_writer:
            metric_writer.scalar("Inner Updates", i)
            metric_writer.scalar("Mean Loss", mean_loss)

        return mean_loss

    def _policy_loss(
        self,
        new_distribution,
        action_batch,
        log_prob_batch_old,
        adv_batch,
        *,
        metric_writer: Writer = None,
    ):
        action_log_probs_new = self.get_log_prob(new_distribution, action_batch)
        ratio = torch.exp(action_log_probs_new - log_prob_batch_old)
        # if ratio explodes to (inf or Nan) due to the residual being to large check initialisation!
        # Generated action probs from (new policy) and (old policy).
        # Values of [0..1] means that actions less likely with the new policy,
        # while values [>1] mean action a more likely now
        clamped_ratio = torch.clamp(
            ratio,
            min=1.0 - self._surrogate_clipping_value,
            max=1.0 + self._surrogate_clipping_value,
        )

        policy_loss = -torch.min(ratio * adv_batch, clamped_ratio * adv_batch).mean()
        entropy_loss = new_distribution.entropy().mean() * self._entropy_reg_coef

        with torch.no_grad():
            approx_kl = (
                (log_prob_batch_old - action_log_probs_new).mean().detach().cpu().item()
            )

        if metric_writer:
            metric_writer.scalar("ratio", ratio.mean().detach().cpu().item())
            metric_writer.scalar("entropy_loss", entropy_loss.detach().cpu().item())
            metric_writer.scalar(
                "clamped_ratio", clamped_ratio.mean().detach().cpu().item()
            )

        return policy_loss - entropy_loss, approx_kl

    def inner_update(self, *transitions, metric_writer: Writer = None) -> Tuple:
        batch_generator = shuffled_batches(
            *transitions, size=transitions[0].size(0), batch_size=self._mini_batch_size
        )
        for (
            state,
            action,
            log_prob_old,
            discounted_signal,
            advantage,
        ) in batch_generator:
            new_distribution, value_estimate = self.actor_critic(state)

            policy_loss, approx_kl = self._policy_loss(
                new_distribution,
                action,
                log_prob_old,
                advantage,
                metric_writer=metric_writer,
            )
            critic_loss = (
                self._critic_criterion(value_estimate, discounted_signal)
                * self._value_reg_coef
            )

            loss = policy_loss + critic_loss

            self._optimiser.zero_grad()
            loss.backward()
            self.post_process_gradients(self.actor_critic.parameters())
            self._optimiser.step()

            if metric_writer:
                metric_writer.scalar(
                    "stddev", new_distribution.stddev.cpu().mean().item()
                )
                metric_writer.scalar("policy_loss", policy_loss.detach().cpu().item())
                metric_writer.scalar("critic_loss", critic_loss.detach().cpu().item())
                metric_writer.scalar("approx_kl", approx_kl)
                metric_writer.scalar("loss", loss.detach().cpu().item())

            if approx_kl > 1.5 * self._target_kl:
                return loss.detach().cpu().item(), True
            return loss.detach().cpu().item(), False
