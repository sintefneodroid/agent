import copy
from typing import Any, Sequence

import numpy
import torch
import torch.nn.functional as F
from numpy import mean
from tqdm import tqdm

from draugr import (
    MockWriter,
    Writer,
    to_tensor,
    frozen_parameters,
    freeze_model,
    frozen_model,
)
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    TransitionPointBuffer,
    TransitionPoint,
    ConcatInputMLP,
    MLP,
)
from neodroidagent.common.architectures.mlp_variants.concatination import (
    PostConcatInputMLP,
)
from neodroidagent.utilities import (
    ActionSpaceNotSupported,
    OrnsteinUhlenbeckProcess,
    update_target,
    is_zero_or_mod_zero,
)
from warg import GDKC, drop_unused_kws, super_init_pass_on_kws

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

    # region Private

    def __init__(
        self,
        random_process_spec=GDKC(constructor=OrnsteinUhlenbeckProcess),
        memory_buffer=TransitionPointBuffer(),
        evaluation_function=F.mse_loss,
        actor_arch_spec=GDKC(MLP, output_activation=torch.nn.Tanh()),
        critic_arch_spec=GDKC(PostConcatInputMLP),
        discount_factor: float = 0.95,
        update_target_interval=1,
        batch_size=128,
        noise_factor=1e-1,
        copy_percentage=0.005,
        actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=1e-4),
        critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=1e-2),
        **kwargs,
    ):
        """

@param random_process_spec:
@param memory_buffer:
@param evaluation_function:
@param actor_arch_spec:
@param critic_arch_spec:
@param discount_factor:
@param update_target_interval:
@param batch_size:
@param noise_factor:
@param copy_percentage:
@param actor_optimiser_spec:
@param critic_optimiser_spec:
@param kwargs:
"""
        super().__init__(**kwargs)

        assert 0 <= discount_factor <= 1.0
        assert 0 <= copy_percentage <= 1.0

        self._copy_percentage = copy_percentage
        self._actor_optimiser_spec: GDKC = actor_optimiser_spec
        self._critic_optimiser_spec: GDKC = critic_optimiser_spec
        self._actor_arch_spec = actor_arch_spec
        self._critic_arch_spec = critic_arch_spec
        self._random_process_spec = random_process_spec

        self._memory_buffer = memory_buffer
        self._critic_criteria = evaluation_function
        self._actor_arch_spec = actor_arch_spec
        self._critic_arch_spec = critic_arch_spec
        self._discount_factor = discount_factor
        self._update_target_interval = update_target_interval

        self._batch_size = batch_size
        self._noise_factor = noise_factor

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
@param critic:
@param critic_optimiser:
@param actor:
@param actor_optimiser:
@return:
"""

        if action_space.is_discrete:
            raise ActionSpaceNotSupported()

        self._actor_arch_spec.kwargs["input_shape"] = self._input_shape
        self._actor_arch_spec.kwargs["output_shape"] = self._output_shape
        self._actor = self._actor_arch_spec().to(self._device)
        self._target_actor = copy.deepcopy(self._actor).to(self._device)
        freeze_model(self._target_actor, True, True)
        self._actor_optimiser = self._actor_optimiser_spec(self._actor.parameters())

        self._critic_arch_spec.kwargs["input_shape"] = (
            *self._input_shape,
            *self._output_shape,
        )
        self._critic_arch_spec.kwargs["output_shape"] = 1
        self._critic = self._critic_arch_spec().to(self._device)
        self._target_critic = copy.deepcopy(self._critic).to(self._device)
        freeze_model(self._target_critic, True, True)
        self._critic_optimiser = self._critic_optimiser_spec(self._critic.parameters())

        self._random_process = self._random_process_spec(
            sigma=mean([r.span for r in action_space.ranges])
        )

    # region Public

    @property
    def models(self):
        """

@return:
"""
        return {"_actor": self._actor, "_critic": self._critic}

    def update_targets(self, update_percentage, *, metric_writer: Writer = None):
        """

@param update_percentage:
@return:
"""
        with torch.no_grad():
            if metric_writer:
                metric_writer.blip("Target Model Synced", self.update_i)

            update_target(
                target_model=self._target_critic,
                source_model=self._critic,
                copy_percentage=update_percentage,
            )
            update_target(
                target_model=self._target_actor,
                source_model=self._actor,
                copy_percentage=update_percentage,
            )

    # endregion

    # region Protected
    @drop_unused_kws
    def _remember(self, *, signal, terminated, state, successor_state, sample):
        self._memory_buffer.add_transition_point(
            TransitionPoint(state, sample, successor_state, signal, terminated)
        )

    @drop_unused_kws
    def _update(self, *, metric_writer: Writer = MockWriter()):
        """
Update

:return:
:rtype:
"""
        tensorised = TransitionPoint(
            *[
                to_tensor(a, device=self._device)
                for a in self._memory_buffer.sample_transition_points()
            ]
        )

        self._memory_buffer.clear()

        # Compute next Q value based on which action target actor would choose
        # Detach variable from the current graph since we don't want gradients for next Q to propagated
        with torch.no_grad():
            next_max_q = self._target_critic(
                tensorised.successor_state, self._target_actor(tensorised.state)
            )
            Q_target = tensorised.signal + (
                self._discount_factor * next_max_q * tensorised.non_terminal_numerical
            )
            # Compute the target of the current Q values

        # Compute current Q value, critic takes state and action chosen
        td_error = self._critic_criteria(
            self._critic(tensorised.state, tensorised.action), Q_target.detach()
        )
        self._critic_optimiser.zero_grad()
        td_error.backward()
        self.post_process_gradients(self._critic.parameters())
        self._critic_optimiser.step()

        with frozen_model(self._critic):
            policy_loss = -torch.mean(
                self._critic(tensorised.state, self._actor(tensorised.state))
            )
            self._actor_optimiser.zero_grad()
            policy_loss.backward()
            self.post_process_gradients(self._actor.parameters())
            self._actor_optimiser.step()

        if is_zero_or_mod_zero(self._update_target_interval, self.update_i):
            self.update_targets(self._copy_percentage, metric_writer=metric_writer)

        if metric_writer:
            metric_writer.scalar("td_error", td_error.cpu().item())
            metric_writer.scalar("critic_loss", policy_loss.cpu().item())

        with torch.no_grad():
            return (td_error + policy_loss).cpu().item()

    def extract_action(self, sample: Any) -> numpy.ndarray:
        """

@param sample:
@return:
"""
        return sample.to("cpu").numpy()

    @drop_unused_kws
    def _sample(self, state: Sequence) -> Any:
        """

@param state:
@param deterministic:
@return:
"""

        with torch.no_grad():
            action_out = self._actor(to_tensor(state, device=self._device)).detach()

        deterministic = False
        if not deterministic:
            # Add action space noise for exploration, alternative is parameter space noise
            noise = self._random_process.sample_transition_points(action_out.shape)
            action_out += to_tensor(noise * self._noise_factor, device=self.device)

        return action_out
        # endregion
