#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "heider"
__doc__ = r"""

           Created on 7/15/22
           """

__all__ = ["CommonEnvironmentScalarEnum", "CommonProcedureScalarEnum"]

from enum import Enum

from sorcery import assigned_names


class CommonEnvironmentScalarEnum(Enum):
    """description"""

    (
        signal,
        running_signal,
        signal_since_last_termination,
        duration,
        duration_since_last_termination,
        running_mean_action,
        fps,
    ) = assigned_names()


class CommonProcedureScalarEnum(Enum):
    """description"""

    (
        new_best_model,
        num_batch_epochs,
        policy_loss,
        policy_std,
        policy_entropy,
        critic_loss,
        td_error,
        loss,
    ) = assigned_names()
