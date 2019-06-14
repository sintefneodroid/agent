#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
__doc__ = ''

from tqdm import tqdm


def step_wise_training(C,
                       agent,
                       environment,
                       *,
                       batch_length,
                       num_updates,
                       num_batches,
                       stat_frequency,
                       render_frequency):
  B = range(1, num_updates + 1)
  B = tqdm(B, f'Batch {0}, {num_batches} - Episode {agent._rollout_i}',
           leave=False)

  initial_state = environment.reset()
  for batch_i in B:
    if batch_i % stat_frequency == 0:
      pass
      # B.set_description(f'Batch {batch_i}, {num_batches} - Episode {agent._rollout_i}')

    if render_frequency and batch_i % render_frequency == 0:
      (transitions,
       accumulated_signal,
       terminated,
       initial_state) = agent.take_n_steps(initial_state,
                                           environment,
                                           render=True,
                                           n=batch_length
                                           )
    else:
      (transitions,
       accumulated_signal,
       terminated,
       initial_state) = agent.take_n_steps(initial_state,
                                           environment,
                                           n=batch_length
                                           )

    if batch_i >= agent._initial_observation_period:
      advantage_memories = agent.back_trace(transitions)
      for m in advantage_memories:
        agent._experience_buffer.add(m)

      agent.update_models()
      agent._experience_buffer.clear()

      if agent._rollout_i % agent._update_target_interval == 0:
        agent._actor_critic_target.load_state_dict(agent._actor_critic.state_dict()
                                                   )

    if agent.end_training:
      break

  return agent._actor_critic, []
