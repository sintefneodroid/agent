#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 15/07/2020
           """

import collections

import akro
import numpy
from garage.misc import tensor_utils


class TrajectoryBatch(
    collections.namedtuple(
        "TrajectoryBatch",
        [
            "env_spec",
            "observations",
            "last_observations",
            "actions",
            "rewards",
            "terminals",
            "env_infos",
            "agent_infos",
            "lengths",
        ],
    )
):
    # pylint: disable=missing-return-doc, missing-return-type-doc, missing-param-doc, missing-type-doc  # noqa: E501
    r"""A tuple representing a batch of whole trajectories.

    Data type for on-policy algorithms.

    A :class:`TrajectoryBatch` represents a batch of whole trajectories
    produced when one or more agents interacts with one or more environments.

    +-----------------------+-------------------------------------------------+
    | Symbol                | Description                                     |
    +=======================+=================================================+
    | :math:`N`             | Trajectory index dimension                      |
    +-----------------------+-------------------------------------------------+
    | :math:`[T]`           | Variable-length time dimension of each          |
    |                       | trajectory                                      |
    +-----------------------+-------------------------------------------------+
    | :math:`S^*`           | Single-step shape of a time-series tensor       |
    +-----------------------+-------------------------------------------------+
    | :math:`N \bullet [T]` | A dimension computed by flattening a            |
    |                       | variable-length time dimension :math:`[T]` into |
    |                       | a single batch dimension with length            |
    |                       | :math:`sum_{i \in N} [T]_i`                     |
    +-----------------------+-------------------------------------------------+

    Attributes:
        env_spec (garage.envs.EnvSpec): Specification for the environment from
            which this data was sampled.
        observations (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T], O^*)` containing the (possibly
            multi-dimensional) observations for all time steps in this batch.
            These must conform to :obj:`env_spec.observation_space`.
        last_observations (numpy.ndarray): A numpy array of shape
            :math:`(N, O^*)` containing the last observation of each
            trajectory.  This is necessary since there are one more
            observations than actions every trajectory.
        actions (numpy.ndarray): A  numpy array of shape
            :math:`(N \bullet [T], A^*)` containing the (possibly
            multi-dimensional) actions for all time steps in this batch. These
            must conform to :obj:`env_spec.action_space`.
        rewards (numpy.ndarray): A numpy array of shape
            :math:`(N \bullet [T])` containing the rewards for all time steps
            in this batch.
        terminals (numpy.ndarray): A boolean numpy array of shape
            :math:`(N \bullet [T])` containing the termination signals for all
            time steps in this batch.
        env_infos (dict): A dict of numpy arrays arbitrary environment state
            information. Each value of this dict should be a numpy array of
            shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
        agent_infos (numpy.ndarray): A dict of numpy arrays arbitrary agent
            state information. Each value of this dict should be a numpy array
            of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
            For example, this may contain the hidden states from an RNN policy.
        lengths (numpy.ndarray): An integer numpy array of shape :math:`(N,)`
            containing the length of each trajectory in this batch. This may be
            used to reconstruct the individual trajectories.

    Raises:
        ValueError: If any of the above attributes do not conform to their
            prescribed types and shapes.

    """
    __slots__ = ()

    def __new__(
        cls,
        env_spec,
        observations,
        last_observations,
        actions,
        rewards,
        terminals,
        env_infos,
        agent_infos,
        lengths,
    ):  # noqa: D102
        # pylint: disable=too-many-branches

        first_observation = observations[0]
        first_action = actions[0]
        inferred_batch_size = lengths.sum()

        # lengths
        if len(lengths.shape) != 1:
            raise ValueError(
                "Lengths tensor must be a tensor of shape (N,), but got a "
                "tensor of shape {} instead".format(lengths.shape)
            )

        if not (lengths.dtype.kind == "u" or lengths.dtype.kind == "i"):
            raise ValueError(
                "Lengths tensor must have an integer dtype, but got dtype {} "
                "instead.".format(lengths.dtype)
            )

        # observations
        if not env_spec.observation_space.contains(first_observation):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(
                env_spec.observation_space, (akro.Box, akro.Discrete, akro.Dict)
            ):
                if env_spec.observation_space.flat_dim != numpy.prod(
                    first_observation.shape
                ):
                    raise ValueError(
                        "observations should have the same "
                        "dimensionality as the observation_space "
                        "({}), but got data with shape {} "
                        "instead".format(
                            env_spec.observation_space.flat_dim, first_observation.shape
                        )
                    )
            else:
                raise ValueError(
                    "observations must conform to observation_space {}, but "
                    "got data with shape {} instead.".format(
                        env_spec.observation_space, first_observation
                    )
                )

        if observations.shape[0] != inferred_batch_size:
            raise ValueError(
                "Expected batch dimension of observations to be length {}, "
                "but got length {} instead.".format(
                    inferred_batch_size, observations.shape[0]
                )
            )

        # observations
        if not env_spec.observation_space.contains(last_observations[0]):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(
                env_spec.observation_space, (akro.Box, akro.Discrete, akro.Dict)
            ):
                if env_spec.observation_space.flat_dim != numpy.prod(
                    last_observations[0].shape
                ):
                    raise ValueError(
                        "last_observations should have the same "
                        "dimensionality as the observation_space "
                        "({}), but got data with shape {} "
                        "instead".format(
                            env_spec.observation_space.flat_dim,
                            last_observations[0].shape,
                        )
                    )
            else:
                raise ValueError(
                    "last_observations must conform to observation_space {}, "
                    "but got data with shape {} instead.".format(
                        env_spec.observation_space, last_observations[0]
                    )
                )

        if last_observations.shape[0] != len(lengths):
            raise ValueError(
                "Expected batch dimension of last_observations to be length "
                "{}, but got length {} instead.".format(
                    len(lengths), last_observations.shape[0]
                )
            )

        # actions
        if not env_spec.action_space.contains(first_action):
            # Discrete actions can be either in the space normally, or one-hot
            # encoded.
            if isinstance(env_spec.action_space, (akro.Box, akro.Discrete, akro.Dict)):
                if env_spec.action_space.flat_dim != numpy.prod(first_action.shape):
                    raise ValueError(
                        "actions should have the same "
                        "dimensionality as the action_space "
                        "({}), but got data with shape {} "
                        "instead".format(
                            env_spec.action_space.flat_dim, first_action.shape
                        )
                    )
            else:
                raise ValueError(
                    "actions must conform to action_space {}, but got data "
                    "with shape {} instead.".format(env_spec.action_space, first_action)
                )

        if actions.shape[0] != inferred_batch_size:
            raise ValueError(
                "Expected batch dimension of actions to be length {}, but got "
                "length {} instead.".format(inferred_batch_size, actions.shape[0])
            )

        # rewards
        if rewards.shape != (inferred_batch_size,):
            raise ValueError(
                "Rewards tensor must have shape {}, but got shape {} "
                "instead.".format(inferred_batch_size, rewards.shape)
            )

        # terminals
        if terminals.shape != (inferred_batch_size,):
            raise ValueError(
                "terminals tensor must have shape {}, but got shape {} "
                "instead.".format(inferred_batch_size, terminals.shape)
            )

        if terminals.dtype != numpy.bool:
            raise ValueError(
                "terminals tensor must be dtype numpy.bool, but got tensor "
                "of dtype {} instead.".format(terminals.dtype)
            )

        # env_infos
        for key, val in env_infos.items():
            if not isinstance(val, (dict, numpy.ndarray)):
                raise ValueError(
                    "Each entry in env_infos must be a numpy array or "
                    "dictionary, but got key {} with value type {} instead.".format(
                        key, type(val)
                    )
                )

            if isinstance(val, numpy.ndarray) and val.shape[0] != inferred_batch_size:
                raise ValueError(
                    "Each entry in env_infos must have a batch dimension of "
                    "length {}, but got key {} with batch size {} instead.".format(
                        inferred_batch_size, key, val.shape[0]
                    )
                )

        # agent_infos
        for key, val in agent_infos.items():
            if not isinstance(val, (dict, numpy.ndarray)):
                raise ValueError(
                    "Each entry in agent_infos must be a numpy array or "
                    "dictionary, but got key {} with value type {} instead."
                    "instead".format(key, type(val))
                )

            if isinstance(val, numpy.ndarray) and val.shape[0] != inferred_batch_size:
                raise ValueError(
                    "Each entry in agent_infos must have a batch dimension of "
                    "length {}, but got key {} with batch size {} instead.".format(
                        inferred_batch_size, key, val.shape[0]
                    )
                )

        return super().__new__(
            TrajectoryBatch,
            env_spec,
            observations,
            last_observations,
            actions,
            rewards,
            terminals,
            env_infos,
            agent_infos,
            lengths,
        )

    @classmethod
    def concatenate(cls, *batches):
        """Create a TrajectoryBatch by concatenating TrajectoryBatches.

        Args:
            batches (list[TrajectoryBatch]): Batches to concatenate.

        Returns:
            TrajectoryBatch: The concatenation of the batches.

        """
        if __debug__:
            for b in batches:
                assert set(b.env_infos.keys()) == set(batches[0].env_infos.keys())
                assert set(b.agent_infos.keys()) == set(batches[0].agent_infos.keys())
        env_infos = {
            k: numpy.concatenate([b.env_infos[k] for b in batches])
            for k in batches[0].env_infos.keys()
        }
        agent_infos = {
            k: numpy.concatenate([b.agent_infos[k] for b in batches])
            for k in batches[0].agent_infos.keys()
        }
        return cls(
            batches[0].env_spec,
            numpy.concatenate([batch.observations for batch in batches]),
            numpy.concatenate([batch.last_observations for batch in batches]),
            numpy.concatenate([batch.actions for batch in batches]),
            numpy.concatenate([batch.rewards for batch in batches]),
            numpy.concatenate([batch.terminals for batch in batches]),
            env_infos,
            agent_infos,
            numpy.concatenate([batch.lengths for batch in batches]),
        )

    def split(self):
        """Split a TrajectoryBatch into a list of TrajectoryBatches.

        The opposite of concatenate.

        Returns:
            list[TrajectoryBatch]: A list of TrajectoryBatches, with one
                trajectory per batch.

        """
        trajectories = []
        start = 0
        for i, length in enumerate(self.lengths):
            stop = start + length
            traj = TrajectoryBatch(
                env_spec=self.env_spec,
                observations=self.observations[start:stop],
                last_observations=numpy.asarray([self.last_observations[i]]),
                actions=self.actions[start:stop],
                rewards=self.rewards[start:stop],
                terminals=self.terminals[start:stop],
                env_infos=tensor_utils.slice_nested_dict(self.env_infos, start, stop),
                agent_infos=tensor_utils.slice_nested_dict(
                    self.agent_infos, start, stop
                ),
                lengths=numpy.asarray([length]),
            )
            trajectories.append(traj)
            start = stop
        return trajectories

    def to_trajectory_list(self):
        """Convert the batch into a list of dictionaries.

        Returns:
            list[dict[str, numpy.ndarray or dict[str, numpy.ndarray]]]: Keys:
                * observations (numpy.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*) (the unflattened state
                    space of the current environment).  observations[i] was
                    used by the agent to choose actions[i].
                * next_observations (numpy.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i].
                * actions (numpy.ndarray): Non-flattened array of actions. Should
                    have shape (T, S^*) (the unflattened action space of the
                    current environment).
                * rewards (numpy.ndarray): Array of rewards of shape (T,) (1D
                    array of length timesteps).
                * dones (numpy.ndarray): Array of dones of shape (T,) (1D array
                    of length timesteps).
                * agent_infos (dict[str, numpy.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, numpy.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

        """
        start = 0
        trajectories = []
        for i, length in enumerate(self.lengths):
            stop = start + length
            trajectories.append(
                {
                    "observations": self.observations[start:stop],
                    "next_observations": numpy.concatenate(
                        (
                            self.observations[1 + start : stop],
                            [self.last_observations[i]],
                        )
                    ),
                    "actions": self.actions[start:stop],
                    "rewards": self.rewards[start:stop],
                    "env_infos": {
                        k: v[start:stop] for (k, v) in self.env_infos.items()
                    },
                    "agent_infos": {
                        k: v[start:stop] for (k, v) in self.agent_infos.items()
                    },
                    "dones": self.terminals[start:stop],
                }
            )
            start = stop
        return trajectories

    @classmethod
    def from_trajectory_list(cls, env_spec, paths):
        """Create a TrajectoryBatch from a list of trajectories.

        Args:
            env_spec (garage.envs.EnvSpec): Specification for the environment
                from which this data was sampled.
            paths (list[dict[str, numpy.ndarray or dict[str, numpy.ndarray]]]): Keys:
                * observations (numpy.ndarray): Non-flattened array of
                    observations. Typically has shape (T, S^*) (the unflattened
                    state space of the current environment). observations[i]
                    was used by the agent to choose actions[i]. observations
                    may instead have shape (T + 1, S^*).
                * next_observations (numpy.ndarray): Non-flattened array of
                    observations. Has shape (T, S^*). next_observations[i] was
                    observed by the agent after taking actions[i]. Optional.
                    Note that to ensure all information from the environment
                    was preserved, observations[i] should have shape (T + 1,
                    S^*), or this key should be set. However, this method is
                    lenient and will "duplicate" the last observation if the
                    original last observation has been lost.
                * actions (numpy.ndarray): Non-flattened array of actions. Should
                    have shape (T, S^*) (the unflattened action space of the
                    current environment).
                * rewards (numpy.ndarray): Array of rewards of shape (T,) (1D
                    array of length timesteps).
                * dones (numpy.ndarray): Array of rewards of shape (T,) (1D array
                    of length timesteps).
                * agent_infos (dict[str, numpy.ndarray]): Dictionary of stacked,
                    non-flattened `agent_info` arrays.
                * env_infos (dict[str, numpy.ndarray]): Dictionary of stacked,
                    non-flattened `env_info` arrays.

        """
        lengths = numpy.asarray([len(p["rewards"]) for p in paths])
        if all(
            len(path["observations"]) == length + 1
            for (path, length) in zip(paths, lengths)
        ):
            last_observations = numpy.asarray([p["observations"][-1] for p in paths])
            observations = numpy.concatenate([p["observations"][:-1] for p in paths])
        else:
            # The number of observations and timesteps must match.
            observations = numpy.concatenate([p["observations"] for p in paths])
            if paths[0].get("next_observations") is not None:
                last_observations = numpy.asarray(
                    [p["next_observations"][-1] for p in paths]
                )
            else:
                last_observations = numpy.asarray(
                    [p["observations"][-1] for p in paths]
                )

        stacked_paths = tensor_utils.concat_tensor_dict_list(paths)
        return cls(
            env_spec=env_spec,
            observations=observations,
            last_observations=last_observations,
            actions=stacked_paths["actions"],
            rewards=stacked_paths["rewards"],
            terminals=stacked_paths["dones"],
            env_infos=stacked_paths["env_infos"],
            agent_infos=stacked_paths["agent_infos"],
            lengths=lengths,
        )

    @staticmethod
    def traj_list_to_tensors(paths, max_path_length, baseline_predictions, discount):
        """Return processed sample data based on the collected paths.

        Args:
            paths (list[dict]): A list of collected paths.
            max_path_length (int): Maximum length of a single rollout.
            baseline_predictions(numpy.ndarray): : Predicted value of GAE
                (Generalized Advantage Estimation) Baseline.
            discount (float): Environment reward discount.

        Returns:
            dict: Processed sample data, with key
                * observations (numpy.ndarray): Padded array of the observations of
                    the environment
                * actions (numpy.ndarray): Padded array of the actions fed to the
                    the environment
                * rewards (numpy.ndarray): Padded array of the acquired rewards
                * agent_infos (dict): a dictionary of {stacked tensors or
                    dictionary of stacked tensors}
                * env_infos (dict): a dictionary of {stacked tensors or
                    dictionary of stacked tensors}
                * rewards (numpy.ndarray): Padded array of the validity information


        """
        baselines = []
        returns = []

        for idx, path in enumerate(paths):
            # baselines
            path["baselines"] = baseline_predictions[idx]
            baselines.append(path["baselines"])

            # returns
            path["returns"] = tensor_utils.discount_cumsum(path["rewards"], discount)
            returns.append(path["returns"])

        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list(
            [tensor_utils.pad_tensor_dict(p, max_path_length) for p in agent_infos]
        )

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list(
            [tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos]
        )

        valids = [numpy.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            agent_infos=agent_infos,
            env_infos=env_infos,
            valids=valids,
        )

        return samples_data


def log_performance(itr, batch, discount, prefix="Evaluation"):
    """Evaluate the performance of an algorithm on a batch of trajectories.

    Args:
        itr (int): Iteration number.
        batch (TrajectoryBatch): The trajectories to evaluate with.
        discount (float): Discount value, from algorithm's property.
        prefix (str): Prefix to add to all logged keys.

    Returns:
        numpy.ndarray: Undiscounted returns.

    """
