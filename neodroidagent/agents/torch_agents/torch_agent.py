#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import ABC, abstractmethod
from pathlib import Path

from neodroid.utilities import ObservationSpace, SignalSpace, ActionSpace
from typing import Dict

from neodroidagent.agents.agent import Agent, ClipFeature
from neodroidagent.common.architectures.architecture import Architecture
import torch
import hashlib
from draugr import (
    save_model,
    load_latest_model,
    sprint,
    TensorBoardPytorchWriter,
    Writer,
    MockWriter,
    global_torch_device,
)

from warg import passes_kws_to, super_init_pass_on_kws, drop_unused_kws

__author__ = "Christian Heider Nielsen"
__doc__ = r"""
          """
__all__ = ["TorchAgent"]


@super_init_pass_on_kws(super_base=Agent)
class TorchAgent(Agent, ABC):
    """

"""

    # region Private

    def __init__(
        self,
        *,
        device: str = global_torch_device(True),
        gradient_clipping: ClipFeature = ClipFeature(False, -1.0, 1.0),
        gradient_norm_clipping: ClipFeature = ClipFeature(False, -1.0, 1.0),
        **kwargs,
    ):
        """

@param device:
@param gradient_clipping:
@param grad_clip_low:
@param grad_clip_high:
@param kwargs:
"""
        super().__init__(**kwargs)
        self._gradient_clipping = gradient_clipping
        self._gradient_norm_clipping = gradient_norm_clipping
        self._device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
        )

    def post_process_gradients(self, model):
        """

    @param model:
    @return:
    """
        if self._gradient_clipping.enabled:
            for params in model.parameters():
                params.grad.data.clamp_(
                    self._gradient_clipping.low, self._gradient_clipping.high
                )

        if self._gradient_norm_clipping.enabled:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self._gradient_norm_clipping.high
            )

    @property
    def device(self) -> torch.device:
        """

@return:
"""
        return self._device

    # endregion

    def build(
        self,
        observation_space: ObservationSpace,
        action_space: ActionSpace,
        signal_space: SignalSpace,
        *,
        metric_writer: Writer = MockWriter(),
        print_model_repr: bool = True,
        **kwargs,
    ) -> None:
        """

    @param observation_space:
    @param action_space:
    @param signal_space:
    @param metric_writer:
    @param print_model_repr:
    @param kwargs:
    @return:
    """
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
                    dummy_in = torch.rand(1, *self.input_shape)

                    model = copy.deepcopy(w)
                    model.to("cpu")
                    if isinstance(metric_writer, TensorBoardPytorchWriter):
                        metric_writer.graph(model, dummy_in)

    # region Public
    @property
    @abstractmethod
    def models(self) -> Dict[str, Architecture]:
        """

@return:
"""
        raise NotImplementedError

    @passes_kws_to(save_model)
    def save(self, **kwargs) -> None:
        """

@param kwargs:
@return:
"""
        for k, v in self.models.items():
            save_model(v, model_name=self.model_name(k, v), **kwargs)

    @staticmethod
    def model_name(k, v) -> str:
        """

@param k:
@param v:
@return:
"""
        model_repr = "".join([str(a) for a in v.named_children()])
        # print(model_repr)
        model_hash = hashlib.md5(model_repr.encode("utf-8")).hexdigest()
        return f"{k}-{model_hash}"

    def on_load(self) -> None:
        pass

    @drop_unused_kws
    def load(self, *, save_directory: Path, evaluation: bool = False) -> None:
        """

@param save_directory:
@param evaluation:
@return:
"""
        loaded = True
        if save_directory.exists():
            print("Loading models froms: " + str(save_directory))
            for k, v in self.models.items():
                model_identifier = self.model_name(k, v)
                latest = load_latest_model(save_directory, model_identifier)
                if latest:
                    model = getattr(self, k)
                    model.load_state_dict(latest)

                    if evaluation:
                        model = model.eval()
                        model.train(False)

                    model = model.to(self._device)

                    setattr(self, k, model)
                else:
                    loaded = False
                    print(f"Missing a model for {model_identifier}")

        if not loaded:
            print("Some models where not found in: " + str(save_directory))

        self.on_load()

    def eval(self) -> None:
        [m.eval() for m in self.models.values()]

    # endregion
