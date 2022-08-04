#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Christian Heider Nielsen"

__all__ = ["OffPolicyBatched"]
__doc__ = "Collects agent experience in a batched fashion for off policy agents"

import logging
from typing import Optional

from draugr.metrics.accumulation import mean_accumulator
from draugr.tqdm_utilities import progress_bar
from draugr.writers import MockWriter, Writer


from neodroidagent.utilities.misc.common_metrics import CommonEnvironmentScalarEnum

from neodroidagent.common.session_factory.vertical.procedures.procedure_specification import (
    Procedure,
)
from warg import is_positive_and_mod_zero
from draugr.drawers import MplDrawer, MockDrawer
from neodroid.utilities import to_one_hot


class OffPolicyBatched(Procedure):
    def __call__(
        self,
        *,
        batch_size: int = 1000,
        iterations: int = 10000,
        stat_frequency: int = 10,
        render_frequency: int = 10,
        disable_stdout: bool = False,
        train_agent: bool = True,
        metric_writer: Optional[Writer] = MockWriter(),
        drawer: MplDrawer = MockDrawer(),
        **kwargs,
    ) -> None:
        """


        :param log_directory:
        :param num_steps:
        :param iterations:
        :param stat_frequency:
        :param render_frequency:
        :param disable_stdout:
        :return:
        :rtype: object
        :param batch_size:
        :param log_directory:
        :param iterations:
        :param stat_frequency:
        :param render_frequency:
        :param disable_stdout:
        :param train_agent:
        :param kwargs:"""

        state = self.agent.extract_features(self.environment.reset())

        running_signal = mean_accumulator()
        best_running_signal = None
        running_mean_action = mean_accumulator()

        for batch_i in progress_bar(
            range(1, iterations),
            disable=disable_stdout,
            description="Batch #",
            postfix=f"Agent update #{self.agent.update_i}",
        ):
            for _ in progress_bar(
                range(batch_size),
                disable=disable_stdout,
                description="Step #",
            ):

                sample = self.agent.sample(state)
                action = self.agent.extract_action(sample)
                snapshot = self.environment.react(action)
                successor_state = self.agent.extract_features(snapshot)
                signal = self.agent.extract_signal(snapshot)

                if is_positive_and_mod_zero(render_frequency, batch_i):
                    self.environment.render()
                    if drawer is not None and action is not None:
                        if self.environment.action_space.is_singular_discrete:
                            action_a = to_one_hot(self.agent.output_shape, action)
                        else:
                            action_a = action[0]
                        drawer.draw(action_a)

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
                metric_writer.scalar(
                    CommonEnvironmentScalarEnum.running_signal.value, sig, batch_i
                )
                metric_writer.scalar(
                    CommonEnvironmentScalarEnum.running_mean_action.value, rma, batch_i
                )

            if train_agent:
                loss = self.agent.update(metric_writer=metric_writer)

                if sig > best_running_signal:
                    best_running_signal = sig
                    self.model_improved(step_i=self.agent.update_i, loss=loss, **kwargs)
            else:
                logging.info("no update")

            if self.early_stop:
                break
