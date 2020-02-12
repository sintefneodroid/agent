#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import Union

import numpy
import torch
from tqdm import tqdm

from draugr.torch_utilities.to_tensor import to_tensor
from draugr.writers import TensorBoardPytorchWriter

__author__ = "Christian Heider Nielsen"

__all__ = ["OffPolicyBatched"]
__doc__ = "Collects agent experience in a batched fashion for off policy agents"

from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from neodroidagent.common.memory import Transition
from neodroidagent.utilities import is_positive_and_mod_zero


class OffPolicyBatched(Procedure):
    def __call__(
        self,
        *,
        num_steps_per_batch=1000,
        device: Union[str, torch.device],
        log_directory: Union[str, Path],
        iterations=10000,
        stat_frequency=10,
        render_frequency=10,
        disable_stdout: bool = False,
        train_agent: bool = True,
        **kwargs
    ) -> None:
        """

:param device:
:param log_directory:
:param num_steps:
:param iterations:
:param stat_frequency:
:param render_frequency:
:param disable_stdout:
:return:
    @rtype: object
    @param num_steps_per_batch:
    @param log_directory:
    @param iterations:
    @param stat_frequency:
    @param render_frequency:
    @param disable_stdout:
    @param train_agent:
    @param kwargs:
    @type device: object
"""
        # with torch.autograd.detect_anomaly():
        with TensorBoardPytorchWriter(log_directory) as metric_writer:

            state = self.agent.extract_features(self.environment.reset())

            best_loss = math.inf

            B = range(1, iterations)
            B = tqdm(B, leave=False, disable=disable_stdout, desc="Batch #")
            for batch_i in B:

                batch_signal = []

                S = range(num_steps_per_batch)
                S = tqdm(S, leave=False, disable=disable_stdout, desc="Step #")
                for _ in S:

                    sample = self.agent.sample(state)
                    snapshot = self.environment.react(self.agent.extract_action(sample))
                    successor_state = self.agent.extract_features(snapshot)
                    signal = self.agent.extract_signal(snapshot)

                    if is_positive_and_mod_zero(render_frequency, batch_i):
                        self.environment.render()

                    batch_signal.append(signal)

                    if train_agent:
                        self.agent.remember(
                            state=state,
                            signal=signal,
                            terminated=snapshot.terminated,
                            sample=sample,
                            successor_state=successor_state,
                        )

                    state = successor_state

                if is_positive_and_mod_zero(stat_frequency, batch_i):
                    metric_writer.scalar(
                        "Batch signal", numpy.sum(batch_signal).item(), batch_i
                    )

                if train_agent:
                    loss = self.agent.update(metric_writer=metric_writer)

                    if best_loss > loss:
                        best_loss = loss
                        self.call_on_improvement_callbacks(loss=loss, **kwargs)
                else:
                    print("no update")

                if self.early_stop:
                    break
