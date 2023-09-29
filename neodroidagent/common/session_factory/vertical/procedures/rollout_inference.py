#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 9/5/19
           """

__all__ = ["RolloutInference"]

from itertools import count
from pathlib import Path
from typing import Union

import torch
from draugr.torch_utilities import TensorBoardPytorchWriter
from draugr.visualisation import progress_bar
from warg import drop_unused_kws

from .procedure_specification import Procedure


class RolloutInference(Procedure):
    @drop_unused_kws
    def __call__(
        self,
        *,
        iterations: int = 9999,
        log_directory: Union[str, Path],
        render_frequency: int = 100,
        stat_frequency: int = 10,
    ):
        """
        :param iterations:
        :param log_directory:
        :param render_frequency:
        :param stat_frequency:
        :return:
        """
        with torch.no_grad():
            with TensorBoardPytorchWriter(log_directory) as metric_writer:
                B = progress_bar(count(), description=f"step {0}, {iterations}")

                for _ in B:
                    initial_state = self.environment.reset()
                    self.agent.rollout(initial_state, self.environment)

                    if self.early_stop:
                        break

        return self.agent.models
