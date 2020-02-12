#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from pathlib import Path
from typing import Union

import numpy
import torch
from tqdm import tqdm

from draugr.writers import TensorBoardPytorchWriter

__author__ = "Christian Heider Nielsen"
__all__ = ["OffPolicyStepWise"]
__doc__ = "Collects agent experience in a step wise fashion"

from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from neodroidagent.utilities import is_positive_and_mod_zero


class OffPolicyStepWise(Procedure):
    def __call__(
        self,
        *,
        num_steps=500000,
        batch_size=128,
        device: Union[str, torch.device],
        log_directory: Union[str, Path],
        stat_frequency=10,
        render_frequency=10,
        initial_observation_period=1000,
        update_agent_frequency: int = 50,
        disable_stdout: bool = False,
        train_agent: bool = True,
        **kwargs
    ) -> None:
        """

:param device:
:param log_directory:
:param num_steps:
:param stat_frequency:
:param render_frequency:
:param disable_stdout:
:return:
"""
        # with torch.autograd.detect_anomaly():
        with TensorBoardPytorchWriter(log_directory) as metric_writer:
            state = self.agent.extract_features(self.environment.reset())

            best_loss = math.inf
            termination_i = 0
            signal_since_last_termination = 0
            duration_since_last_termination = 0

            for step_i in tqdm(range(num_steps), desc="Step #", leave=False):

                sample = self.agent.sample(state)
                snapshot = self.environment.react(self.agent.extract_action(sample))
                successor_state = self.agent.extract_features(snapshot)
                signal = self.agent.extract_signal(snapshot)
                terminated = snapshot.terminated

                if train_agent:
                    self.agent.remember(
                        state=state,
                        signal=signal,
                        terminated=terminated,
                        sample=sample,
                        successor_state=successor_state,
                    )

                state = successor_state

                duration_since_last_termination += 1
                signal_since_last_termination += signal.mean().item()

                if (
                    train_agent
                    and is_positive_and_mod_zero(update_agent_frequency, step_i)
                    and len(self.agent.memory) > batch_size
                ):
                    loss = self.agent.update(
                        batch_size=batch_size, metric_writer=metric_writer
                    )

                    if best_loss > loss:
                        best_loss = loss
                        self.call_on_improvement_callbacks(loss=loss, **kwargs)

                if terminated.any():
                    termination_i += 1
                    if metric_writer:
                        metric_writer.scalar(
                            "duration_since_last_termination",
                            duration_since_last_termination,
                        )
                        metric_writer.scalar(
                            "signal_since_last_termination",
                            signal_since_last_termination,
                        )
                    signal_since_last_termination = 0
                    duration_since_last_termination = 0

                if is_positive_and_mod_zero(stat_frequency, termination_i):
                    self.environment.render()

                if self.early_stop:
                    break


if __name__ == "__main__":
    sw = OffPolicyStepWise()
