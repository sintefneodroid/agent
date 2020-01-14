#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Union, Any

import numpy
import torch

from draugr.writers import TensorBoardPytorchWriter
from neodroid.environments.unity_environment import VectorUnityEnvironment
from neodroid.utilities import EnvironmentSnapshot
from neodroidagent.common.procedures.procedure_specification import Procedure
from neodroidagent.common.transitions import ValuedTransition
from neodroidagent.utilities.bool_tests import is_set_mod_zero_ret_alt
from warg.kw_passing import drop_unused_kws, passes_kws_to

__author__ = "Christian Heider Nielsen"
__doc__ = ""

from tqdm import tqdm


@drop_unused_kws
def take_n_steps(
    agent,
    initial_state: EnvironmentSnapshot,
    environment: VectorUnityEnvironment,
    n: int = 100,
    *,
    render: bool = False,
) -> Any:
    state = initial_state.observables

    accumulated_signal = []

    snapshot = None
    transitions = []
    terminated = False
    value_estimates = []
    T = tqdm(range(1, n + 1), desc=f"Step #", leave=False, disable=not render)
    for t in T:
        action, action_prob, value_estimates, *_ = agent.sample(state)

        snapshot = environment.react(action)

        successor_state, signal, terminated = (
            snapshot.observables,
            snapshot.signal,
            snapshot.terminated,
        )

        transitions.append(
            ValuedTransition(
                state,
                action,
                signal,
                successor_state,
                terminated,
                action_prob,
                value_estimates,
            )
        )

        state = successor_state

        accumulated_signal.append(signal)

        if numpy.array(terminated).all():
            # TODO: support individual reset of environments vector
            snapshot = environment.reset()
            state, signal, terminated = (
                snapshot.observables,
                snapshot.signal,
                snapshot.terminated,
            )

    transitions = ValuedTransition(*zip(*transitions))

    return transitions, accumulated_signal, terminated, snapshot, value_estimates


class StepWise2(Procedure):
    @drop_unused_kws
    @passes_kws_to(take_n_steps)
    def __call__(
        self,
        *,
        num_steps_per_btach: int = 256,
        num_updates: int = 10,
        iterations: int = 9999,
        log_directory: Union[str, Path],
        render_frequency: int = 100,
        stat_frequency: int = 10,
        train_agent: bool = True,
        **kwargs,
    ):
        """

:param num_steps_per_btach:
:param num_updates:
:param iterations:
:param log_directory:
:param render_frequency:
:param stat_frequency:
:return:
"""
        with torch.autograd.detect_anomaly():
            with TensorBoardPytorchWriter(log_directory) as metric_writer:
                initial_state = self.environment.reset()

                B = range(1, num_updates + 1)
                B = tqdm(B, f"Batch {0}, {iterations}", leave=False)

                for batch_i in B:
                    a = take_n_steps(
                        self.agent,
                        initial_state,
                        self.environment,
                        render=is_set_mod_zero_ret_alt(render_frequency, batch_i),
                        metric_writer=is_set_mod_zero_ret_alt(
                            stat_frequency, batch_i, ret=metric_writer
                        ),
                        n=num_steps_per_btach,
                        **kwargs,
                    )
                    (
                        transitions,
                        accumulated_signal,
                        terminated,
                        initial_state,
                        next_value,
                    ) = a

                    if train_agent:
                        self.agent.update(transitions, next_value)
                    else:
                        print("no update")

                    if self.early_stop:
                        break
