#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from draugr import sprint
from draugr.torch_utilities import GraphWriterMixin
from draugr.torch_utilities import (
    get_model_hash,
    global_torch_device,
    load_latest_model_parameters,
    save_model_parameters,
    Architecture,
)
from draugr.writers import MockWriter, Writer
from neodroidagent.agents.agent import Agent, TogglableLowHigh
from neodroidagent.utilities import IntrinsicSignalProvider
from neodroidagent.utilities.exploration.intrinsic_signals.braindead import (
    BraindeadIntrinsicSignalProvider,
)
from torch.nn import Parameter
from torch.optim import Optimizer
from trolls.spaces import ActionSpace, ObservationSpace, SignalSpace
from warg import drop_unused_kws, passes_kws_to, super_init_pass_on_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
          """
__all__ = ["TorchAgent"]


@super_init_pass_on_kws(super_base=Agent)
class TorchAgent(Agent, ABC):
    """ """

    # region Private

    def __init__(
        self,
        *,
        device: str = global_torch_device(True),
        gradient_clipping: TogglableLowHigh = TogglableLowHigh(False, -1.0, 1.0),
        gradient_norm_clipping: TogglableLowHigh = TogglableLowHigh(False, -1.0, 1.0),
        intrinsic_signal_provider_arch: IntrinsicSignalProvider = BraindeadIntrinsicSignalProvider,
        **kwargs,
    ):
        """

        :param device:
        :param gradient_clipping:
        :param grad_clip_low:
        :param grad_clip_high:
        :param kwargs:"""
        super().__init__(
            intrinsic_signal_provider_arch=intrinsic_signal_provider_arch, **kwargs
        )
        self._gradient_clipping = gradient_clipping
        self._gradient_norm_clipping = gradient_norm_clipping
        self._device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

    def post_process_gradients(self, parameters: Iterable[Parameter]) -> None:
        """

        :param model:
        :return:
            :param parameters:"""
        if self._gradient_clipping.enabled:
            for params in parameters:
                params.grad.data.clamp_(
                    self._gradient_clipping.low, self._gradient_clipping.high
                )

        if self._gradient_norm_clipping.enabled:
            torch.nn.utils.clip_grad_norm_(
                parameters, self._gradient_norm_clipping.high
            )

    @property
    def device(self) -> torch.device:
        """

        :return:"""
        return self._device

    # endregion

    def build(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        *,
        metric_writer: Optional[Writer] = MockWriter(),
        print_model_repr: bool = True,
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """

        :param observation_space:
        :param action_space:
        :param signal_space:
        :param metric_writer:
        :param print_model_repr:
        :param kwargs:
        :return:
            :param verbose:"""
        super().build(
            observation_space,
            action_space,
            signal_space,
            print_model_repr=print_model_repr,
            metric_writer=metric_writer,
            **kwargs,
        )

        if print_model_repr:
            for k, w in self.models.items():
                sprint(f"{k}: {w}", highlight=True, color="cyan")

                if metric_writer:
                    try:
                        model = copy.deepcopy(w).to("cpu")
                        dummy_input = model.sample_input()
                        sprint(f"{k} input: {dummy_input.shape}")

                        import contextlib

                        with contextlib.redirect_stdout(
                            None
                        ):  # So much useless frame info printed... Suppress it
                            if isinstance(metric_writer, GraphWriterMixin):
                                metric_writer.graph(
                                    model, dummy_input, verbose=verbose
                                )  # No naming available at moment...
                    except RuntimeError as ex:
                        sprint(
                            f"Tensorboard(Pytorch) does not support you model! No graph added: {str(ex).splitlines()[0]}",
                            color="red",
                            highlight=True,
                        )

    # region Public
    @property
    @abstractmethod
    def models(self) -> Dict[str, Architecture]:
        """

        :return:"""
        raise NotImplementedError

    @property
    @abstractmethod
    def optimisers(self) -> Dict[str, Optimizer]:
        """

        :return:"""
        raise NotImplementedError

    @passes_kws_to(save_model_parameters)
    def save(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:"""
        for (k, v), o in zip(self.models.items(), self.optimisers.values()):
            save_model_parameters(
                v, optimiser=o, model_name=self.model_name(k, v), **kwargs
            )

    @staticmethod
    def model_name(k, v) -> str:
        """

        :param k:
        :param v:
        :return:"""
        return f"{k}-{get_model_hash(v)}"

    def on_load(self) -> None:
        pass

    @drop_unused_kws
    def load(self, *, save_directory: Path, evaluation: bool = False) -> bool:
        """

        :param save_directory:
        :param evaluation:
        :return:"""
        loaded = True
        if save_directory.exists():
            print(f"Loading models from: {str(save_directory)}")
            for (model_key, model), (optimiser_key, optimiser) in zip(
                self.models.items(), self.optimisers.items()
            ):
                model_identifier = self.model_name(model_key, model)
                (model, optimiser), loaded = load_latest_model_parameters(
                    model,
                    model_name=model_identifier,
                    optimiser=optimiser,
                    model_directory=save_directory,
                )
                if loaded:
                    model = model.to(self._device)
                    # optimiser = optimiser.to(self._device)
                    if evaluation:
                        model = model.eval()
                        model.train(False)  # Redundant
                    setattr(self, model_key, model)
                    setattr(self, optimiser_key, optimiser)
                else:
                    loaded = False
                    print(f"Missing a model for {model_identifier}")

        if not loaded:
            print(f"Some models where not found in: {str(save_directory)}")

        self.on_load()

        return loaded

    def eval(self) -> None:
        [m.eval() for m in self.models.values()]

    def train(self) -> None:
        [m.train() for m in self.models.values()]

    # endregion
