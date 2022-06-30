#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional

from draugr.drawers import MockDrawer, MplDrawer
from draugr.metrics.accumulation import mean_accumulator
from draugr.writers import MockWriter, Writer
from tqdm import tqdm

__author__ = "Christian Heider Nielsen"
__all__ = ["OffPolicyStepWise"]
__doc__ = "Collects agent experience in a step wise fashion"

from neodroid.utilities import to_one_hot
from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from warg import is_positive_and_mod_zero, is_zero_or_mod_below


class OffPolicyStepWise(Procedure):
    def __call__(
        self,
        *,
        num_environment_steps=500000,
        batch_size=128,
        stat_frequency=10,
        render_frequency=10000,
        initial_observation_period=1000,
        render_duration=1000,
        update_agent_frequency: int = 1,
        disable_stdout: bool = False,
        train_agent: bool = True,
        metric_writer: Optional[Writer] = MockWriter(),
        drawer: MplDrawer = MockDrawer(),
        **kwargs
    ) -> None:
        """

        :param log_directory:
        :param num_environment_steps:
        :param stat_frequency:
        :param render_frequency:
        :param disable_stdout:
        :return:"""

        state = self.agent.extract_features(self.environment.reset())

        running_signal = mean_accumulator()
        best_running_signal = None
        running_mean_action = mean_accumulator()
        termination_i = 0
        signal_since_last_termination = 0
        duration_since_last_termination = 0

        it = tqdm(range(num_environment_steps), desc="Step #", leave=False)
        for step_i in it:

            sample = self.agent.sample(state)
            action = self.agent.extract_action(sample)

            snapshot = self.environment.react(action)
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
            mean_signal = signal.mean().item()
            signal_since_last_termination += mean_signal

            running_mean_action.send(action.mean())
            running_signal.send(mean_signal)

            if (
                train_agent
                and is_positive_and_mod_zero(
                    update_agent_frequency * batch_size, step_i
                )
                and len(self.agent.memory_buffer) > batch_size
                and step_i > initial_observation_period
            ):
                loss = self.agent.update(metric_writer=metric_writer)

                sig = next(running_signal)
                if not best_running_signal or sig > best_running_signal:
                    best_running_signal = sig
                    self.model_improved(
                        step_i=self.agent.update_i, loss=loss, signal=sig, **kwargs
                    )

            if terminated.any():
                termination_i += 1
                if metric_writer:
                    metric_writer.scalar(
                        "duration_since_last_termination",
                        duration_since_last_termination,
                    )
                    metric_writer.scalar(
                        "signal_since_last_termination", signal_since_last_termination
                    )
                    metric_writer.scalar(
                        "running_mean_action", next(running_mean_action)
                    )
                    metric_writer.scalar("running_signal", next(running_signal))
                signal_since_last_termination = 0
                duration_since_last_termination = 0

            if (
                is_zero_or_mod_below(
                    render_frequency,
                    render_duration,
                    step_i,
                    residual_printer=it.status_printer,
                )
                and render_frequency != 0
            ):
                self.environment.render()
                if drawer is not None and action is not None:
                    if self.environment.action_space.is_singular_discrete:
                        action_a = to_one_hot(self.agent.output_shape, action)
                    else:
                        action_a = action[0]
                    drawer.draw(action_a)

            if self.early_stop:
                break


if __name__ == "__main__":
    sw = OffPolicyStepWise()
