#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

from abc import ABC, abstractmethod

import numpy as np

from neodroidagent.utilities.misc.environment_model import env_stats


class NumpyAgent(ABC):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.parameters = {}
        self.hyperparameters = {}
        self.derived_variables = {}
        self.env_info = env_stats(env)
        self.episode_history = {"rewards": [], "state_actions": []}

    def _create_2num_dicts(self, obs_encoder=None, act_encoder=None):
        E = self.env_info
        n_states = np.prod(E["n_obs_per_dim"])
        n_actions = np.prod(E["n_actions_per_dim"])

        # create action -> scalar dictionaries
        self._num2action = dict()
        self._action2num = dict(act_encoder)
        if n_actions != np.inf:
            self._action2num = {act: i for i, act in enumerate(E["action_ids"])}
            self._num2action = {i: act for act, i in self._action2num.items()}

        # create obs -> scalar dictionaries
        self._num2obs = dict()
        self._obs2num = dict(obs_encoder)
        if n_states != np.inf:
            self._obs2num = {act: i for i, act in enumerate(E["obs_ids"])}
            self._num2obs = {i: act for act, i in self._obs2num.items()}

    def flush_history(self) -> None:
        """Clear the episode history"""
        for k, v in self.episode_history.items():
            self.episode_history[k] = []

    @abstractmethod
    def sample(self, obs):
        raise NotImplementedError

    @abstractmethod
    def greedy_policy(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def run_episode(self, max_steps, render=False):
        raise NotImplementedError

    @abstractmethod
    def update(self):
        raise NotImplementedError
