#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import draugr
from agent.interfaces.specifications import TR
from neodroid.interfaces.specifications import EnvironmentSnapshot

__author__ = 'cnheider'
__doc__ = ''

from tqdm import tqdm


def step_wise_training(agent,
                       environment,
                       *,

                       batch_length=100,
                       num_updates=10,
                       num_batches=9999,
                       log_directory: Union[str, Path],
                       render_frequency=100,
                       stat_frequency=10,

**kwargs
                       ) -> TR:
  B = range(1, num_updates + 1)
  B = tqdm(B, f'Batch {0}, {num_batches}',
           leave=False)

  initial_state = environment.reset()

  with draugr.TensorBoardXWriter(str(log_directory)) as stat_writer:
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

        agent.update()
        agent._experience_buffer.clear()

        if agent._rollout_i % agent._update_target_interval == 0:
          agent._update_targets()

      if agent.end_training:
        break

  return TR(agent.models, [])
