#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
          """
__all__ = ["TorchAgent"]

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from draugr.torch_utilities import (
    Architecture,
    GraphWriterMixin,
    get_model_hash,
    global_torch_device,
    load_latest_model_parameters,
    save_model_parameters,
)
from draugr.writers import MockWriter, Writer
from neodroidagent.agents.agent import Agent, TogglableLowHigh, TogglableValue
from neodroidagent.utilities import IntrinsicSignalProvider
from neodroidagent.utilities.exploration.intrinsic_signals.braindead import (
    BraindeadIntrinsicSignalProvider,
)
from torch import nn
from torch.nn import Parameter
from torch.optim import Optimizer
from trolls.spaces import ActionSpace, ObservationSpace, SignalSpace
from draugr.python_utilities import sprint
from warg import drop_unused_kws, passes_kws_to, super_init_pass_on_kws


@super_init_pass_on_kws(super_base=Agent)
class TorchAgent(Agent, ABC):
    """ """

    # region Private

    def __init__(
        self,
        *,
        device: str = global_torch_device(True),
        gradient_clipping: TogglableLowHigh = TogglableLowHigh(False, -1.0, 1.0),
        gradient_norm_clipping: TogglableValue = TogglableValue(False, 1.0),
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

    def post_process_gradients(
        self,
        parameters: Iterable[Parameter],
        *,
        metric_writer: Optional[Writer] = MockWriter(),
        parameter_set_name: str = None,
    ) -> None:
        """
        :param parameters:
        :type parameters:
        :param metric_writer:
        :type metric_writer:
        :param parameter_set_name:
        :type parameter_set_name:
        :return:"""

        trainable_parameters = [
            p for p in parameters if p.grad is not None and p.requires_grad
        ]

        total_norm2_pre = 0
        if metric_writer:
            for p in trainable_parameters:
                total_norm2_pre += p.grad.detach().data.norm(2).item() ** 2
            total_norm2_pre = total_norm2_pre**0.5

        if self._gradient_clipping.enabled:
            for params in trainable_parameters:
                params.grad.data.clamp_(
                    self._gradient_clipping.low, self._gradient_clipping.high
                )  # INPLACE OPERATION

        if self._gradient_norm_clipping.enabled:
            total_norm = torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                self._gradient_norm_clipping.value,
                norm_type=2.0,
                error_if_nonfinite=True,
            )  # INPLACE OPERATION

        if metric_writer:
            total_norm2_post = 0
            for p in trainable_parameters:
                total_norm2_post += p.grad.detach().data.norm(2).item() ** 2.0
            total_norm2_post = total_norm2_post**0.5

            if parameter_set_name is None:
                parameter_set_name = "model_parameters"
            metric_writer.scalar(
                f"{parameter_set_name}_grad_norm2_pre_process", total_norm2_pre
            )
            metric_writer.scalar(
                f"{parameter_set_name}_grad_norm2_post_process", total_norm2_post
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

                if isinstance(w, nn.Module):
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
    def optimisers(self) -> Dict[str, Dict[str, Optimizer]]:
        """

        :return:"""
        raise NotImplementedError

    @passes_kws_to(save_model_parameters)
    def save(self, **kwargs) -> None:
        """

        :param kwargs:
        :return:"""
        for k, v in self.models.items():
            o = None
            if k in self.optimisers:
                o = next(iter(self.optimisers[k].values()))

            save_model_parameters(
                v, optimiser=o, model_name=self.model_name(k, v), **kwargs
            )

    @staticmethod
    def model_name(k, v) -> str:
        """

        :param k:
        :param v:
        :return:"""
        if not isinstance(v, nn.Module):
            return f"{k}"
        return f"{k}-{get_model_hash(v)}"

    def on_load(self) -> None:
        """

        :param self:
        :type self:
        :return:
        :rtype:
        """
        pass

    @drop_unused_kws
    def load(self, *, save_directory: Path, evaluation: bool = False) -> bool:
        """

        :param self:
        :type self:
        :param save_directory:
        :param evaluation:
        :return:"""
        loaded = True
        if save_directory.exists():
            print(f"Loading models from: {str(save_directory)}")
            for model_key, model in self.models.items():
                print(f"Loading model: {model_key}")
                model_identifier = self.model_name(model_key, model)
                optimiser_key, optimiser = next(
                    iter(self.optimisers[model_key].items())
                )
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
                    print(
                        f"Loaded model: {model_identifier} {model_key} {optimiser_key}"
                    )
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
        [m.eval() for m in self.models.values() if isinstance(m, nn.Module)]

    def train(self) -> None:
        [m.train() for m in self.models.values() if isinstance(m, nn.Module)]


# endregion
