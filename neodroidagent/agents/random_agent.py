from typing import Any, Optional

from draugr.writers import MockWriter, Writer

from neodroid.utilities.specifications.unity_specifications import EnvironmentSnapshot
from neodroidagent.agents.agent import Agent
from trolls.spaces import ActionSpace, ObservationSpace, SignalSpace

__all__ = ["RandomAgent"]


class RandomAgent(Agent):
    def eval(self) -> None:
        pass

    def _update(
        self, *args, metric_writer: Optional[Writer] = MockWriter(), **kwargs
    ) -> None:
        pass

    def _sample(
        self,
        state: EnvironmentSnapshot,
        *args,
        deterministic: bool = False,
        metric_writer: Optional[Writer] = MockWriter(),
        **kwargs
    ) -> Any:
        """

        :param state:
        :param args:
        :param deterministic:
        :param metric_writer:
        :param kwargs:
        :return:"""
        self._sample_i_since_last_update += 1
        return self.action_space.sample()

    def _remember(self, *, signal, **kwargs):
        pass

    # region Private

    def __build__(
        self,
        *,
        observation_space: ObservationSpace = None,
        action_space: ActionSpace = None,
        signal_space: SignalSpace = None,
        **kwargs
    ) -> None:
        """

        :param observation_space:
        :param action_space:
        :param signal_space:
        :param kwargs:
        :return:"""
        self.action_space = action_space

    def models(self):
        return {}

    # endregion

    # region Public

    def evaluate(self, batch, *args, **kwargs) -> Any:
        pass

    def load(self, *args, **kwargs) -> None:
        pass

    def save(self, *args, **kwargs) -> None:
        pass

    # endregion
