#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from itertools import count
from pathlib import Path
from typing import Union

import torch
from tqdm import tqdm

from draugr.writers import TensorBoardPytorchWriter
from neodroidagent.utilities.specifications.training_resume import TR
from neodroidagent.utilities.specifications.procedure_specification import Procedure

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 9/5/19
           '''


class RolloutInference(Procedure):
  def __call__(self,
               *,
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
    with torch.no_grad():
      with TensorBoardPytorchWriter(log_directory) as metric_writer:

        B = tqdm(count(),
                 f'step {0}, {iterations}',
                 leave=False)

        for _ in B:
          initial_state = self.environment.reset()
          self.agent.rollout(initial_state, self.environment)

          if self.early_stop:
            break

    return TR(self.agent.models, [])
