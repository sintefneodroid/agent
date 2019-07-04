#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm

import draugr
from agent.interfaces.specifications import TR, ValuedTransition
from agent.utilities import to_tensor
from neodroid.interfaces.environment_models import EnvironmentSnapshot

__author__ = 'cnheider'
__doc__ = ''


def batched_training(agent,
                     environment,
                     *,
                     device,
                     log_directory,
                     num_steps=200,
                     rollouts=10000,
                     stat_frequency=100,
                     render_frequency=100,
                     disable_stdout=False,
                     **kwargs
                     ) -> TR:
  with draugr.TensorBoardXWriter(str(log_directory)) as stat_writer:
    state = environment.reset()

    if isinstance(state, EnvironmentSnapshot):
      state = state.observables


    B = range(1, rollouts)
    B = tqdm(B, leave=False, disable=disable_stdout)
    for i in B:
      if agent.end_training:
        break

      batch_signal = []
      transitions = []

      state = to_tensor(state, device=device)
      successor_state = None

      S = range(num_steps)
      S = tqdm(S, leave=False, disable=disable_stdout)
      for _ in S:

        action, action_log_prob, value_estimate, *_ = agent.sample_action(state)

        successor_state, signal, terminated, *_ = environment.step(action)

        if render_frequency and i % render_frequency == 0:
          environment.render()

        if stat_frequency and i % stat_frequency == 0:
          pass
          # stat_writer.scalar()

        batch_signal.append(signal)

        successor_state = to_tensor(successor_state, device=device)
        signal_ = to_tensor(signal, device=device)
        terminated = to_tensor(terminated, device=device)

        transitions.append(ValuedTransition(state,
                                            action,
                                            signal_,
                                            successor_state,
                                            terminated,
                                            value_estimate,
                                            action_log_prob,
                                            )
                           )

        state = successor_state

        # if i % test_interval == 0:
        # test_signal, *_ = agent.rollout(successor_state, environment, render=True)

        # if test_signal > agent._solved_threshold and agent._early_stop:
        #  agent.end_training = True

      # stats.batch_signal.append(batch_signal)

      # only calculate value of next state for the last step this time
      *_, agent._last_value_estimate, _ = agent._sample_model(successor_state)

      batch = ValuedTransition(*zip(*transitions))

      if len(batch) > 100:
        agent.transitions = batch
        agent.update_models()

  return TR(agent.models, None)


'''
def train_episodically(self,
                       env,
                       rollouts=10000,
                       render=False,
                       render_frequency=1000,
                       stat_frequency=10,
                       **kwargs):

  self._rollout_i = 1

  initial_state = env.reset()

  B = tqdm(range(1, rollouts + 1), f'Batch {0}, {rollouts} - Rollout {self._rollout_i}', leave=False,
           disable=not render)
  for batch_i in B:
    if self.end_training or batch_i > rollouts:
      break

    if batch_i % stat_frequency == 0:
      B.set_description(f'Batch {batch_i}, {rollouts} - Rollout {self._rollout_i}')

    if render and batch_i % render_frequency == 0:
      transitions, accumulated_signal, terminated, *_ = self.rollout(initial_state,
                                                                     env,
                                                                     render=render
                                                                     )
    else:
      transitions, accumulated_signal, terminated, *_ = self.rollout(initial_state,
                                                                     env
                                                                     )

    initial_state = env.reset()

    if batch_i >= self._initial_observation_period:
      advantage_memories = self.back_trace_advantages(transitions)

      for memory in advantage_memories:
        self._memory_buffer._add(memory)

      self.update()
      self._memory_buffer.clear()

      if self._rollout_i % self._update_target_interval == 0:
        self.update_target(target_model=self._target_actor, source_model=self._actor,
                           target_update_tau=self._target_update_tau)
        self.update_target(target_model=self._target_critic, source_model=self._critic,
                           target_update_tau=self._target_update_tau)

    if self.end_training:
      break

  return self._actor, self._critic, []
'''
