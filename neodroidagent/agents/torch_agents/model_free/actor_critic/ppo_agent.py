#!/usr/local/bin/python
# coding: utf-8
from typing import Sequence, Union

import torch
import torch.nn.functional as F
from numpy import mean
from torch import nn

from draugr import batched_recycle
from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import MockWriter
from draugr.writers.writer import Writer
from neodroid.utilities.spaces import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common.architectures import MLP
from neodroidagent.common.architectures.experimental.merged import ConcatInputMLP
from neodroidagent.common.memory import TransitionBuffer
from neodroidagent.common.procedures.training.experimental.step_wise2 import StepWise2
from neodroidagent.common.sessions import ParallelSession, session_entry_point
from neodroidagent.common.transitions import (
    AdvantageDiscountedTransition,
    ValuedTransition,
)
from neodroidagent.utilities.exceptions.exceptions import ActionSpaceNotSupported
from neodroidagent.utilities.exploration import OrnsteinUhlenbeckProcess
from neodroidagent.utilities.signal.advantage_estimation import torch_compute_gae
from neodroidagent.utilities.signal.experimental.discounting import (
    discount_signal_torch,
)
from warg.gdkc import GDKC
from warg.kw_passing import drop_unused_kws, super_init_pass_on_kws

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
        discount_factor=0.95,
        gae_tau=0.15,
        actor_lr=4e-4,
        critic_lr=4e-4,
        entropy_reg_coef=1e-2,
        value_reg_coef=5e-1,
        mini_batches=32,
        update_target_interval=1000,
        max_grad_norm=0.5,
        solved_threshold=-200,
        test_interval=1000,
        early_stop=False,
        rollouts=10000,
        surrogate_clipping_value=3e-1,
        ppo_optimisation_epochs=6,
        state_type=torch.float,
        value_type=torch.float,
        action_type=torch.long,
        exploration_epsilon_start=0.99,
        exploration_epsilon_end=0.05,
        exploration_epsilon_decay=10000,
        copy_percentage=3e-3,
        signal_clipping=False,
        action_clipping=False,
        memory_buffer=TransitionBuffer(),
        actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        actor_arch_spec=GDKC(MLP),
        critic_arch_spec=GDKC(ConcatInputMLP),
        random_process_spec=GDKC(constructor=OrnsteinUhlenbeckProcess),
        **kwargs,
    ) -> None:
        """

    :param discount_factor:
    :param gae_tau:
    :param actor_lr:
    :param critic_lr:
    :param entropy_reg_coef:
    :param value_reg_coef:
    :param mini_batches:
    :param copy_percentage:
    :param update_target_interval:
    :param max_grad_norm:
    :param solved_threshold:
    :param test_interval:
    :param early_stop:
    :param rollouts:
    :param surrogate_clipping_value:
    :param ppo_optimisation_epochs:
    :param state_type:
    :param value_type:
    :param action_type:
    :param exploration_epsilon_start:
    :param exploration_epsilon_end:
    :param exploration_epsilon_decay:
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

        self._discount_factor = discount_factor
        self._gae_tau = gae_tau
        # self._reached_horizon_penalty = -10.
        self._actor_lr = actor_lr
        self._critic_lr = critic_lr
        self._entropy_reg_coef = entropy_reg_coef
        self._value_reg_coef = value_reg_coef
        self._mini_batches = mini_batches
        self._target_update_tau = copy_percentage
        self._update_target_interval = update_target_interval
        self._max_grad_norm = max_grad_norm
        self._solved_threshold = solved_threshold
        self._test_interval = test_interval
        self._early_stop = early_stop
        self._rollouts = rollouts
        self._surrogate_clipping_value = surrogate_clipping_value
        self._ppo_optimisation_epochs = ppo_optimisation_epochs
        self._state_type = state_type
        self._value_type = value_type
        self._action_type = action_type
        # TODO: ExplorationSpec
        # params for epsilon greedy
        self._exploration_epsilon_start = exploration_epsilon_start
        self._exploration_epsilon_end = exploration_epsilon_end
        self._exploration_epsilon_decay = exploration_epsilon_decay

        (
            self._actor,
            self._target_actor,
            self._critic,
            self._target_critic,
            self._actor_optimiser,
            self._critic_optimiser,
        ) = (None, None, None, None, None, None)

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

    @property
    def models(self):
        return {"_actor": self._actor, "_critic": self._critic}

    # region Protected

    @drop_unused_kws
    def _optimise(self, cost):

        self._actor_optimiser.zero_grad()
        self._critic_optimiser.zero_grad()
        cost.backward(retain_graph=True)

        if self._max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self._actor.parameters(), self._max_grad_norm)
            nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)

        self._actor_optimiser.step()
        self._critic_optimiser.step()

    @drop_unused_kws
    def _sample(self, state):
        """
    continuous
    randomly sample from normal distribution, whose mean and variance come from policy network.
    [batch, action_size]

    :param state:
    :type state:
    :param continuous:
    :type continuous:
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """

        model_input = to_tensor(state, device=self._device, dtype=self._state_type)

        distribution = self._actor(model_input)

        with torch.no_grad():
            action = distribution.sample().detach()

        value_estimate = self._critic(model_input, action)

        action_log_prob = distribution.log_prob(action)

        action_n = action.to("cpu").numpy()

        return (action_n, action_log_prob, value_estimate, distribution)

    def _update_targets(self) -> None:
        self._update_target(
            target_model=self._target_actor,
            source_model=self._actor,
            copy_percentage=self._target_update_tau,
        )

        self._update_target(
            target_model=self._target_critic,
            source_model=self._critic,
            copy_percentage=self._target_update_tau,
        )

    @drop_unused_kws
    def _update(
        self, transitions, next_value, metric_writer: Writer = MockWriter()
    ) -> None:

        adv_trans = self.back_trace_advantages(transitions, next_value)

        loader = batched_recycle(adv_trans, n=len(adv_trans) // self._mini_batches)
        for i, mini_batch in zip(range(self._ppo_optimisation_epochs), loader):
            a = AdvantageDiscountedTransition(*zip(*mini_batch))
            loss, new_log_probs, old_log_probs = self.evaluate(a)
            self._optimise(loss)
            if metric_writer:
                print(loss.detach().cpu().numpy())
                metric_writer.scalar("loss", loss.detach().cpu().numpy())

        if self._update_i % self._update_target_interval == 0:
            self._update_targets()

    # endregion

    # region Public

    def back_trace_advantages(
        self, transitions: ValuedTransition, next_value
    ) -> Sequence[AdvantageDiscountedTransition]:

        value_estimates = to_tensor(transitions.value_estimate, device=self._device)
        sig = to_tensor(transitions.signal, device=self._device)
        value_estimates = value_estimates.view(value_estimates.shape[0], -1)
        nt = to_tensor(transitions.non_terminal, device=self._device)
        next_value = to_tensor(next_value, device=self.device).view(1, -1)

        advantages = torch_compute_gae(
            signals=sig,
            values=value_estimates,
            non_terminals=nt,
            discount_factor=self._discount_factor,
            tau=self._gae_tau,
            next_value=next_value,
        )

        discounted_signal = discount_signal_torch(
            sig, self._discount_factor, self.device
        )

        advantage_memories = []
        for i, step in enumerate(zip(*transitions)):
            step = ValuedTransition(*step)
            advantage_memories.append(
                AdvantageDiscountedTransition(
                    step.state,
                    step.action,
                    step.successor_state,
                    step.terminal,
                    step.action_prob,
                    value_estimates[i],
                    discounted_signal[i],
                    advantages[i],
                )
            )

        return advantage_memories

    @drop_unused_kws
    def _remember(self, *, state, action, signal, successor_state, terminated):
        self._memory.add_transition(state, action, signal, successor_state, terminated)

    @drop_unused_kws
    def evaluate(self, batch: AdvantageDiscountedTransition, discrete: bool = False):

        states = to_tensor(batch.state, device=self._device)
        value_estimates = to_tensor(batch.value_estimate, device=self._device)
        advantages = to_tensor(batch.advantage, device=self._device)
        discounted_returns = to_tensor(batch.discounted_return, device=self._device)
        action_log_probs_old = to_tensor(batch.action_prob, device=self._device)

        *_, action_log_probs_new, distribution = self._sample(states)
        action_log_probs_new = to_tensor(action_log_probs_new[0], device=self._device)

        if discrete:
            actions = to_tensor(batch.action, device=self._device)
            action_log_probs_old = action_log_probs_old.gather(-1, actions)
            action_log_probs_new = action_log_probs_new.gather(-1, actions)

        ratio = (action_log_probs_new - action_log_probs_old).exp().unsqueeze(-1)
        # Generated action probs from (new policy) and (old policy).
        # Values of [0..1] means that actions less likely with the new policy,
        # while values [>1] mean action a more likely now
        surrogate = ratio * advantages
        clamped_ratio = torch.clamp(
            ratio,
            min=1.0 - self._surrogate_clipping_value,
            max=1.0 + self._surrogate_clipping_value,
        )
        surrogate_clipped = clamped_ratio * advantages  # (L^CLIP)

        policy_loss = torch.min(surrogate, surrogate_clipped).mean()
        entropy_loss = distribution.entropy().mean() * self._entropy_reg_coef
        policy_loss -= entropy_loss

        value_error = (
            F.mse_loss(value_estimates, discounted_returns) * self._value_reg_coef
        )
        collective_cost = policy_loss + value_error

        return (collective_cost, policy_loss, value_error)

    # endregion


# region Test
def ppo_test():
    ppo_run(environment_type="gym")


def ppo_run(
    rollouts=None, skip: bool = True, environment_type: Union[bool, str] = True
):
    from . import ppo_test_config as C

    if rollouts:
        C.ROLLOUTS = rollouts

    session_entry_point(
        PPOAgent,
        C,
        session=ParallelSession(
            procedure=StepWise2,
            environment_name=C.ENVIRONMENT_NAME,
            auto_reset_on_terminal_state=True,
            environment_type=environment_type,
        ),
        parse_args=False,
        skip_confirmation=skip,
    )


if __name__ == "__main__":
    ppo_test()
# ppo_run()

# endregion
