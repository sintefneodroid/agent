import torch
from draugr.torch_utilities import DisjunctMLP

__all__ = ["DuelingQMLP"]


class DuelingQMLP(DisjunctMLP):
    def forward(self, *act, **kwargs) -> torch.tensor:
        """

        :param act:
        :type act:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        advantages, value = super().forward(*act, **kwargs)
        return value + (advantages - advantages.mean())
