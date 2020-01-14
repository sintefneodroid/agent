#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union

import torch
from tqdm import tqdm

from draugr.writers import TensorBoardPytorchWriter

__author__ = "Christian Heider Nielsen"
__all__ = ["StepWise"]
__doc__ = "Collects agent experience in a step wise fashion"

from neodroidagent.common.procedures.procedure_specification import Procedure
from neodroidagent.utilities.bool_tests import is_set_mod_zero_ret_alt

from warg import drop_unused_kws


class StepWise(Procedure):
    @drop_unused_kws
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
        disable_stdout: bool = False,
        train_agent: bool = True,
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
            T = tqdm(range(num_steps), desc="Step #", leave=False)

            ss = self.environment.reset()
            state = ss.observables
            cumula_signal = 0
            duration = 0
            episode_i = 0
            for a in T:

                if a >= initial_observation_period:
                    action = self.agent.sample(state)
                else:
                    action = self.environment.action_space.sample()

                ss = self.environment.react(action)

                (next_state, signal, terminal, *_) = (
                    ss.observables,
                    ss.signal,
                    ss.terminated,
                )

                duration += 1
                cumula_signal += signal

                self.agent.remember(
                    state=state,
                    action=action,
                    signal=signal,
                    next_state=next_state,
                    terminal=terminal,
                )

                state = next_state

                if train_agent and len(self.agent._memory) > batch_size:
                    self.agent.update(batch_size=batch_size)

                if is_set_mod_zero_ret_alt(stat_frequency, episode_i):
                    self.environment.render()

                if terminal:
                    ss = self.environment.reset()
                    state = ss.observables
                    if metric_writer:
                        metric_writer.scalar("duration", duration, episode_i)
                        metric_writer.scalar("signal", cumula_signal, episode_i)
                    episode_i += 1
                    T.set_postfix_str(f"Episode #{episode_i}")
                    cumula_signal = 0
                    duration = 0

                if self.early_stop:
                    break


if __name__ == "__main__":
    sw = StepWise(None)
