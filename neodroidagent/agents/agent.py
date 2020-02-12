#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Sequence, Tuple

import numpy

from draugr import MockWriter, Writer, sprint
from neodroid.utilities import (
    ActionSpace,
    ObservationSpace,
    SignalSpace,
    EnvironmentSnapshot,
)

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
            Base class for all Neodroid Agents
          """

__all__ = ["Agent"]

ClipFeature = namedtuple("ClipFeature", ("enabled", "low", "high"))


class Agent(ABC):
    """
All agents should inherit from this class
"""

    # region Private

    def __init__(
        self,
        input_shape: Sequence = None,
        output_shape: Sequence = None,
        divide_by_zero_safety: float = 1e-6,
        action_clipping: ClipFeature = ClipFeature(False, -1.0, 1.0),
        signal_clipping: ClipFeature = ClipFeature(False, -1.0, 1.0),
        **kwargs,
    ):
        self._sample_i = 0
        self._update_i = 0
        self._sample_i_since_last_update = 0
        if not hasattr(self, "_memory"):
            self._memory_buffer = None

        self._input_shape = input_shape
        self._output_shape = output_shape

        self._action_clipping = action_clipping
        self._signal_clipping = signal_clipping

        self._divide_by_zero_safety = divide_by_zero_safety

        self.__set_protected_attr(**kwargs)

    def meta_vars(self) -> dict:
        return {"sample_i": self.sample_i, "update_i": self.update_i}

    def set_meta_vars(self, sample_i, update_i) -> None:
        self._sample_i = sample_i
        self._update_i = update_i

    @property
    def update_i(self) -> int:
        return self._update_i

    @property
    def sample_i(self) -> int:
        return self._sample_i

    @property
    def memory(self) -> Any:
        return self._memory_buffer

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def __set_protected_attr(self, **kwargs) -> None:
        for k, v in kwargs.items():
            k_lowered = f'_{k.lstrip("_").lower()}'
            self.__setattr__(k_lowered, v)

    def __infer_io_shapes(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        print_inferred_io_shapes: bool = True,
    ) -> None:
        """
    Tries to infer input and output size from env if either _input_shape or _output_shape, is None or -1 (int)

    :rtype: object
    """

        if self._input_shape is None or self._input_shape == -1:
            self._input_shape = observation_space.shape

        if self._output_shape is None or self._output_shape == -1:
            self._output_shape = action_space.shape

        # region print

        if print_inferred_io_shapes:
            sprint(
                f"input shape: {self._input_shape}\n"
                f"observation space: {observation_space}\n",
                color="green",
                bold=True,
                highlight=True,
            )

            sprint(
                f"output shape: {self._output_shape}\n"
                f"action space: {action_space}\n",
                color="yellow",
                bold=True,
                highlight=True,
            )

            sprint(
                f"signal shape: {signal_space}\n",
                color="blue",
                bold=True,
                highlight=True,
            )

    # endregion

    # region Public

    # @passes_kws_to(__build__)
    def build(self, observation_space, action_space, signal_space, **kwargs) -> None:
        """

    @param observation_space:
    @param action_space:
    @param signal_space:
    @param kwargs:
    @return:
    """
        self.__infer_io_shapes(observation_space, action_space, signal_space)
        self.__build__(
            observation_space=observation_space,
            action_space=action_space,
            signal_space=signal_space,
            **kwargs,
        )

    @property
    def input_shape(self) -> [int, ...]:
        """

    @return:
    """
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: [int, ...]):
        """

    @param input_shape:
    @return:
    """
        self._input_shape = input_shape

    @property
    def output_shape(self) -> [int, ...]:
        """

    @return:
    """
        return self._output_shape

    @output_shape.setter
    def output_shape(self, output_shape: Tuple[int, ...]):
        """

    @param output_shape:
    @return:
    """
        self._output_shape = output_shape

    def sample(
        self,
        state: EnvironmentSnapshot,
        *args,
        deterministic: bool = False,
        metric_writer: Writer = MockWriter(),
        **kwargs,
    ) -> Tuple[Any, ...]:
        """

    @param state:
    @param args:
    @param deterministic:
    @param metric_writer:
    @param kwargs:
    @return:
    """
        self._sample_i += 1
        self._sample_i_since_last_update += 1
        action = self._sample(
            state,
            *args,
            deterministic=deterministic,
            metric_writer=metric_writer,
            **kwargs,
        )

        if self._action_clipping.enabled:
            action = numpy.clip(
                action, self._action_clipping.low, self._action_clipping.high
            )

        return action

    def extract_features(self, snapshot: EnvironmentSnapshot) -> numpy.ndarray:
        """
    Feature extraction
    """

        return numpy.array(snapshot.observables)

    def extract_action(self, sample: Any) -> numpy.ndarray:
        """

    @param sample:
    @return:
    """
        return numpy.array(sample)

    def extract_signal(self, snapshot: EnvironmentSnapshot, **kwargs) -> numpy.ndarray:
        """
    Allows for modulation of signal based on for example an Instrinsic Curiosity signal

    @param signal:
    @param kwargs:
    @return:
    """
        return numpy.array(snapshot.signal)

    def eval(self) -> None:
        """
    @return:
    """
        pass

    def update(self, *args, metric_writer: Writer = MockWriter(), **kwargs) -> float:
        """

    @param args:
    @param metric_writer:
    @param kwargs:
    @return:
    """
        self._update_i += 1
        self._sample_i_since_last_update = 0
        return self._update(*args, metric_writer=metric_writer, **kwargs)

    def remember(self, *, signal: Any, terminated: Any, **kwargs):
        """

    @param terminated:
    @param signal:
    @param kwargs:
    @return:
    """

        if self._signal_clipping.enabled:
            signal = numpy.clip(
                signal, self._signal_clipping.low, self._signal_clipping.high
            )

        self._remember(signal=signal, terminated=terminated, **kwargs)

    # endregion

    # region Abstract

    @abstractmethod
    def __build__(
        self,
        *,
        observation_space: ObservationSpace = None,
        action_space: ActionSpace = None,
        signal_space: SignalSpace = None,
        **kwargs,
    ) -> None:
        """

    @param observation_space:
    @param action_space:
    @param signal_space:
    @param kwargs:
    @return:
    """
        raise NotImplementedError

    @abstractmethod
    def load(self, *, save_directory, **kwargs) -> None:
        """

    @param save_directory:
    @param kwargs:
    @return:
    """
        raise NotImplementedError

    @abstractmethod
    def save(self, *, save_directory, **kwargs) -> None:
        """

    @param save_directory:
    @param kwargs:
    @return:
    """
        raise NotImplementedError

    @abstractmethod
    def _remember(self, *, signal, terminated, **kwargs) -> None:
        """

    @param kwargs:
    @return:
    """
        raise NotImplementedError

    @abstractmethod
    def _sample(
        self,
        state: EnvironmentSnapshot,
        *args,
        deterministic: bool = False,
        metric_writer: Writer = MockWriter(),
        **kwargs,
    ) -> Tuple[Any, ...]:
        """

    @param state:
    @param args:
    @param deterministic:
    @param metric_writer:
    @param kwargs:
    @return:
    """
        raise NotImplementedError

    @abstractmethod
    def _update(self, *args, metric_writer: Writer = MockWriter(), **kwargs) -> Any:
        """

    @param args:
    @param metric_writer:
    @param kwargs:
    @return:
    """
        raise NotImplementedError

    # endregion
