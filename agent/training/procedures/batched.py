#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm import tqdm

from agent.interfaces.specifications import TR, ValuedTransition
from agent.utilities import to_tensor

__author__ = 'cnheider'
__doc__ = ''


def batched_training(C,
                     agent,
                     environment,
                     *,
                     num_steps=200,
                     rollouts=10000,
                     render=True
                     ) -> TR:
  state = environment.reset()
  state = state.observables

  B = range(1, rollouts)
  B = tqdm(B, leave=False, disable=not render)
  for i in B:
    if agent._end_training:
      break

    batch_signal = []
    transitions = []

    state = to_tensor(state, device=C.DEVICE)
    successor_state = None

    S = range(num_steps)
    S = tqdm(S, leave=False, disable=not render)
    for _ in S:

      action, action_log_prob, value_estimate,*_ = agent.sample_action(state)

      successor_state, signal, terminated, *_ = environment.step(action)

      batch_signal.append(signal)

      successor_state = to_tensor(successor_state, device=C.DEVICE)
      signal_ = to_tensor(signal, device=C.DEVICE)
      terminated = to_tensor(terminated, device=C.DEVICE)

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

      agent._step_i += 1

      if agent._step_i % agent._test_interval == 0:
        test_signal,*_ = agent.rollout(successor_state, environment,render=render)

        if test_signal > agent._solved_threshold and agent._early_stop:
          agent._end_training = True

    # stats.batch_signal.append(batch_signal)

    # only calculate value of next state for the last step this time
    *_, agent._last_value_estimate, _ = agent._sample_model(successor_state)

    batch = ValuedTransition(*zip(*transitions))

    if len(batch) > 100:
      agent.evaluate(batch, agent._last_value_estimate)


  return TR((agent._actor, agent._critic), None)




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
    if self._end_training or batch_i > rollouts:
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

    if self._end_training:
      break

  return self._actor, self._critic, []
'''
