#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import torch
from tqdm import tqdm

from draugr.metrics.accumulation import mean_accumulator
from draugr.writers import TensorBoardPytorchWriter

__author__ = "Christian Heider Nielsen"

__all__ = ["OffPolicyBatched"]
__doc__ = "Collects agent experience in a batched fashion for off policy agents"

from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from neodroidagent.utilities import is_positive_and_mod_zero


class OffPolicyBatched(Procedure):
    def __call__(
        self,
        *,
        batch_size=1000,
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
@param batch_size:
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

            running_signal = mean_accumulator()
            best_running_signal = None
            running_mean_action = mean_accumulator()

            for batch_i in tqdm(
                range(1, iterations),
                leave=False,
                disable=disable_stdout,
                desc="Batch #",
            ):
                for _ in tqdm(
                    range(batch_size),
                    leave=False,
                    disable=disable_stdout,
                    desc="Step #",
                ):

                    sample = self.agent.sample(state)
                    action = self.agent.extract_action(sample)
                    snapshot = self.environment.react(action)
                    successor_state = self.agent.extract_features(snapshot)
                    signal = self.agent.extract_signal(snapshot)

                    if is_positive_and_mod_zero(render_frequency, batch_i):
                        self.environment.render()

                    if train_agent:
                        self.agent.remember(
                            state=state,
                            signal=signal,
                            terminated=snapshot.terminated,
                            sample=sample,
                            successor_state=successor_state,
                        )

                    state = successor_state

                    running_signal.send(signal.mean())
                    running_mean_action.send(action.mean())

                sig = next(running_signal)
                rma = next(running_mean_action)

                if is_positive_and_mod_zero(stat_frequency, batch_i):
                    metric_writer.scalar("Running signal", sig, batch_i)
                    metric_writer.scalar("running_mean_action", rma, batch_i)

                if train_agent:
                    loss = self.agent.update(metric_writer=metric_writer)

                    if sig > best_running_signal:
                        best_running_signal = sig
                        self.call_on_improvement_callbacks(loss=loss, **kwargs)
                else:
                    print("no update")

                if self.early_stop:
                    break
