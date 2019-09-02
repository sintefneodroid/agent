#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import torch

from draugr.writers import TensorBoardPytorchWriter
from neodroid.environments.unity.vector_unity_environment import VectorUnityEnvironment

from neodroidagent.interfaces.specifications import TR
from neodroidagent.interfaces.torch_agent import TorchAgent

__author__ = 'Christian Heider Nielsen'
__doc__ = ''

from tqdm import tqdm


def step_wise_training(agent: TorchAgent,
                       environment: VectorUnityEnvironment,
                       *,
                       num_steps_per_btach: int = 256,
                       num_updates: int = 10,
                       num_batches: int = 9999,
                       log_directory: Union[str, Path],
                       render_frequency: int = 100,
                       stat_frequency: int = 10,
                       **kwargs
                       ) -> TR:
  with torch.autograd.detect_anomaly():
    with TensorBoardPytorchWriter(str(log_directory)) as metric_writer:
      B = range(1, num_updates + 1)
      B = tqdm(B, f'Batch {0}, {num_batches}',
               leave=False)

      initial_state = environment.reset()
      for batch_i in B:
        (transitions,
         accumulated_signal,
         terminated,
         initial_state) = agent.take_n_steps(initial_state,
                                             environment,
                                             render=(True
                                                     if (render_frequency and
                                                         batch_i % render_frequency == 0)
                                                     else False),
                                             metric_writer=(metric_writer
                                                            if (stat_frequency and
                                                                batch_i % stat_frequency == 0)
                                                            else
                                                            None),
                                             n=num_steps_per_btach
                                             )

        agent.update(transitions)

        if agent.end_training:
          break

  return TR(agent.models, [])
