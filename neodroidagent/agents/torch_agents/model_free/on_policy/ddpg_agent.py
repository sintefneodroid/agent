import copy
from typing import Any, Sequence

import numpy
import torch
import torch.nn.functional as F
from numpy import mean
from tqdm import tqdm

from draugr import MockWriter, Writer, to_tensor
from neodroid.utilities import ActionSpace, ObservationSpace, SignalSpace
from neodroidagent.agents.torch_agents.torch_agent import TorchAgent
from neodroidagent.common import (
    SingleHeadMLP,
    SingleHeadConcatInputMLP,
    TransitionPointBuffer,
    Transition,
    TransitionPoint,
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
        actor_arch_spec=GDKC(SingleHeadMLP),
        critic_arch_spec=GDKC(SingleHeadConcatInputMLP),
        discount_factor=0.99,
        sync_target_model_frequency=1000,
        batch_size=64,
        noise_factor=4e-1,
        copy_percentage=1e-2,
        actor_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        critic_optimiser_spec: GDKC = GDKC(constructor=torch.optim.Adam, lr=3e-4),
        **kwargs,
    ):
        """

    @param random_process_spec:
    @param memory_buffer:
    @param evaluation_function:
    @param actor_arch_spec:
    @param critic_arch_spec:
    @param discount_factor:
    @param sync_target_model_frequency:
    @param batch_size:
    @param noise_factor:
    @param copy_percentage:
    @param actor_optimiser_spec:
    @param critic_optimiser_spec:
    @param kwargs:
    """
        super().__init__(**kwargs)

        self._target_update_tau = copy_percentage
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
        self._sync_target_model_frequency = sync_target_model_frequency

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
        *,
        critic=None,
        critic_optimiser=None,
        actor=None,
        actor_optimiser=None,
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
        self._target_actor = copy.deepcopy(self._actor).to(self._device).eval()

        self._critic = self._critic_arch_spec().to(self._device)
        self._target_critic = copy.deepcopy(self._critic).to(self._device).eval()

        self._random_process = self._random_process_spec(
            sigma=mean([r.span for r in action_space.ranges])
        )

        # Construct the optimizers for actor and critic
        self._actor_optimiser = self._actor_optimiser_spec(self._actor.parameters())
        self._critic_optimiser = self._critic_optimiser_spec(self._critic.parameters())

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
    def _remember(
        self, *, signal, terminated, state, successor_state, sample, **kwargs
    ):

        self._memory_buffer.add_transition_points(
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
            )  # Compute the target of the current Q values

        # Compute current Q value, critic takes state and action chosen

        td_error = self._critic_criteria(
            self._critic(tensorised.state, tensorised.action), Q_target
        )  # Compute Bellman error (using Huber loss)
        self._critic_optimiser.zero_grad()
        td_error.backward()
        self.post_process_gradients(model=self._critic)
        self._critic_optimiser.step()  # Optimize the critic

        policy_loss = -self._critic(
            tensorised.state, self._actor(tensorised.state)
        ).mean()
        self._actor_optimiser.zero_grad()
        policy_loss.backward()
        self.post_process_gradients(model=self._actor)
        self._actor_optimiser.step()  # Optimize the actor

        if is_zero_or_mod_zero(self._sync_target_model_frequency, self.update_i):
            self.update_targets(self._target_update_tau, metric_writer=metric_writer)

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
        action_out = sample.to("cpu").numpy()
        return action_out

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
            noise = self._random_process.sample(action_out.shape)
            action_out += to_tensor(noise * self._noise_factor, device=self.device)

        return action_out
        # endregion
