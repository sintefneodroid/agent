#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import torch

from draugr.writers import TensorBoardPytorchWriter
from neodroidagent.utilities.specifications import TR, Procedure

__author__ = 'Christian Heider Nielsen'
__doc__ = ''

from tqdm import tqdm


class StepWise(Procedure):

  def __call__(self,
               *,
               num_steps_per_btach: int = 256,
               num_updates: int = 10,
               iterations: int = 9999,
               log_directory: Union[str, Path],
               render_frequency: int = 100,
               stat_frequency: int = 10,
               **kwargs
               ) -> TR:
    '''

    :param num_steps_per_btach:
    :param num_updates:
    :param iterations:
    :param log_directory:
    :param render_frequency:
    :param stat_frequency:
    :param kwargs:
    :return:
    '''
    with torch.autograd.detect_anomaly():
      with TensorBoardPytorchWriter(str(log_directory)) as metric_writer:
        initial_state = self.environment.reset()

        B = range(1, num_updates + 1)
        B = tqdm(B,
                 f'Batch {0}, {iterations}',
                 leave=False)

        for batch_i in B:
          (transitions,
           accumulated_signal,
           terminated,
           initial_state) = self.agent.take_n_steps(initial_state,
                                                    self.
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

          self.agent.update(transitions)

          if self.early_stop:
            break
