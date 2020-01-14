#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from neodroidagent.agents.agent import Agent
from neodroidagent.common.architectures.architecture import Architecture
import torch
import hashlib
from draugr import (
    save_model,
    load_latest_model,
    drop_unused_kws,
    sprint,
    TensorBoardPytorchWriter,
    Writer,
    MockWriter,
    global_torch_device,
)

from warg import passes_kws_to, super_init_pass_on_kws

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
        gradient_clipping=False,
        grad_clip_low=-1.0,
        grad_clip_high=1.0,
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
        self._grad_clip_low = grad_clip_low
        self._grad_clip_high = grad_clip_high
        self._device = torch.device(
            device if torch.cuda.is_available() and device != "cpu" else "cpu"
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
        observation_space,
        action_space,
        signal_space,
        *,
        metric_writer: Writer = MockWriter(),
        print_model_repr=True,
        **kwargs,
    ) -> None:
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
        return f'{k}-{hashlib.md5(f"{v}".encode("utf-8")).hexdigest()}'

    def on_load(self):
        pass

    @drop_unused_kws
    def load(self, model_path: Path, evaluation: bool = False, **kwargs) -> None:
        """

    @param model_path:
    @param evaluation:
    @param kwargs:
    @return:
    """
        if model_path.exists():
            print("Loading models froms: " + str(model_path))
            for k, v in self.models.items():
                latest = load_latest_model(model_path, self.model_name(k, v))
                if latest:
                    model = getattr(self, k)
                    model.load_state_dict(latest)

                    if evaluation:
                        model = model.eval()
                        model.train(False)

                    model = model.to(self._device)

                    setattr(self, k, model)

        self.on_load()

    # endregion
