#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

from collections import defaultdict

import numpy as np

from neodroidagent.agents.numpy_agents.numpy_agent import NumpyAgent
from neodroidagent.utilities.misc.environment_model import tile_state_space


class TemporalDifferenceAgent(NumpyAgent):
    def __init__(
        self,
        env,
        lr=0.4,
        epsilon=0.1,
        n_tilings=8,
        obs_max=None,
        obs_min=None,
        grid_dims=[8, 8],
        off_policy=False,
        temporal_discount=0.99,
    ):
        """
A temporal difference learning agent with expected SARSA (on-policy) or
TD(0) `Q`-learning (off-policy) updates.

Notes
-----
The agent requires a discrete action space, but will try to discretize
the observation space via tiling if it is continuous.

Parameters
----------
env : gym.wrappers or gym.envs instance
The environment to run the agent on.
lr : float
Learning rate for the Q function updates. Default is 0.05.
epsilon : float between [0, 1]
The epsilon value in the epsilon-soft policy. Larger values
encourage greater exploration during training. Default is 0.1.
n_tilings : int
The number of overlapping tilings to use if the ``env`` observation
space is continuous. Unused if observation space is discrete.
Default is 8.
obs_max : float or :py:class:`ndarray <numpy.ndarray>`
The value to treat as the max value of the observation space when
calculating the grid widths if the observation space is continuous.
If None, use ``env.observation_space.high``. Unused if observation
space is discrete. Default is None.
obs_min : float or :py:class:`ndarray <numpy.ndarray>`
The value to treat as the min value of the observation space when
calculating grid widths if the observation space is continuous. If
None, use ``env.observation_space.low``. Unused if observation
space is discrete. Default is None.
grid_dims : list
The number of rows and columns in each tiling grid if the env
observation space is continuous. Unused if observation space is
discrete. Default is [8, 8].
off_policy : bool
Whether to use a behavior policy separate from the target policy
during training. If False, use the same epsilon-soft policy for
both behavior and target policies. Default is False.
temporal_discount : float between [0, 1]
The discount factor used for downweighting future rewards. Smaller
values result in greater discounting of future rewards. Default is
0.9.
"""
        super().__init__(env)

        self.lr = lr
        self.obs_max = obs_max
        self.obs_min = obs_min
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.grid_dims = grid_dims
        self.off_policy = off_policy
        self.temporal_discount = temporal_discount

        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"

        obs_encoder = None
        if E["continuous_observations"]:
            obs_encoder, _ = tile_state_space(
                self.env,
                self.env_info,
                self.n_tilings,
                state_action=False,
                obs_max=self.obs_max,
                obs_min=self.obs_min,
                grid_size=self.grid_dims,
            )

        self._create_2num_dicts(obs_encoder=obs_encoder)

        # behavior policy is stochastic, epsilon-soft policy
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy
        if self.off_policy:
            # target policy is deterministic, greedy policy
            self.target_policy = self._greedy

        # initialize Q function
        self.parameters["Q"] = defaultdict(np.random.rand)

        # initialize returns object for each state-action pair
        self.derived_variables = {"episode_num": 0}

        self.hyperparameters = {
            "agent": "TemporalDifferenceAgent",
            "lr": self.lr,
            "obs_max": self.obs_max,
            "obs_min": self.obs_min,
            "epsilon": self.epsilon,
            "n_tilings": self.n_tilings,
            "grid_dims": self.grid_dims,
            "off_policy": self.off_policy,
            "temporal_discount": self.temporal_discount,
        }

        self.episode_history = {"state_actions": [], "rewards": []}

    def run_episode(self, max_steps, render=False):
        """
Run the agent on a single episode without updating the priority queue
or performing backups.

Parameters
----------
max_steps : int
The maximum number of steps to run an episode
render : bool
Whether to render the episode during training

Returns
-------
reward : float
The total reward on the episode, averaged over the theta samples.
steps : float
The total number of steps taken on the episode, averaged over the
theta samples.
"""
        return self._episode(max_steps, render, update=False)

    def train_episode(self, max_steps, render=False):
        """
Train the agent on a single episode.

Parameters
----------
max_steps : int
The maximum number of steps to run an episode.
render : bool
Whether to render the episode during training.

Returns
-------
reward : float
The total reward on the episode.
steps : float
The number of steps taken on the episode.
"""
        D = self.derived_variables
        total_rwd, n_steps = self._episode(max_steps, render, update=True)

        D["episode_num"] += 1

        return total_rwd, n_steps

    def _episode(self, max_steps, render, update=True):
        """
Run or train the agent on an episode.

Parameters
----------
max_steps : int
The maximum number of steps to run the episode.
render : bool
Whether to render the episode during training.
update : bool
Whether to perform the Q function backups after each step. Default
is True.

Returns
-------
reward : float
The total reward on the episode.
steps : float
The number of steps taken on the episode.
"""
        self.flush_history()

        obs = self.env.reset()
        HS = self.episode_history

        action = self.sample(obs)
        s = self._obs2num[obs]
        a = self._action2num[action]

        # store initial (state, action) tuple
        HS["state_actions"].append((s, a))

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            # take action
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # record rewards
            HS["rewards"].append(reward)
            total_reward += reward

            # generate next state and action
            action = self.sample(obs)
            s_ = self._obs2num[obs] if not done else None
            a_ = self._action2num[action]

            # store next (state, action) tuple
            HS["state_actions"].append((s_, a_))

            if update:
                self.update()

            if done:
                break

        return total_reward, n_steps

    def _epsilon_soft_policy(self, s, a=None):
        """
Epsilon-soft exploration policy.

In epsilon-soft policies, :math:`\pi(a|s) > 0` for all s ∈ S and all a ∈ A(s) at
the start of training. As learning progresses, pi gradually shifts
closer and closer to a deterministic optimal policy.

In particular, we have:

pi(a|s) = 1 - epsilon + (epsilon / |A(s)|) IFF a == a*
pi(a|s) = epsilon / |A(s)|                 IFF a != a*

where

|A(s)| is the number of actions available in state s
a* ∈ A(s) is the greedy action in state s (i.e., a* = argmax_a Q(s, a))

Note that epsilon-greedy policies are instances of epsilon-soft
policies, defined as policies for which pi(a|s) >= epsilon / |A(s)| for
all states and actions.

Parameters
----------
s : int, float, or tuple
The state number for the current observation, as returned by
``self._obs2num[obs]``
a : int, float, or tuple
The action number in the current state, as returned by
self._action2num[obs]. If None, sample an action from the action
probabilities in state s, otherwise, return the probability of
action `a` under the epsilon-soft policy. Default is None.

Returns
-------
If `a` is None:
action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by `self._num2action`
If `a` is None, returns an action sampled from the distribution
over actions defined by the epsilon-soft policy.

If `a` is not None:
action_prob : float in range [0, 1]
If `a` is not None, returns the probability of `a` under the
epsilon-soft policy.
"""
        E, P = self.env_info, self.parameters

        # TODO: this assumes all actions are available in every state
        n_actions = np.prod(E["n_actions_per_dim"])

        a_star = np.argmax([P["Q"][(s, aa)] for aa in range(n_actions)])
        p_a_star = 1.0 - self.epsilon + (self.epsilon / n_actions)
        p_a = self.epsilon / n_actions

        action_probs = np.ones(n_actions) * p_a
        action_probs[a_star] = p_a_star
        np.testing.assert_allclose(np.sum(action_probs), 1)

        if a is not None:
            return action_probs[a]

        # sample action
        a = np.random.multinomial(1, action_probs).argmax()
        return self._num2action[a]

    def _greedy(self, s, a=None):
        """
A greedy behavior policy. Only used when off-policy is true.

Parameters
----------
s : int, float, or tuple
The state number for the current observation, as returned by
``self._obs2num[obs]``
a : int, float, or tuple
The action number in the current state, as returned by
``self._action2num[obs]``. If None, sample an action from the
action probabilities in state `s`, otherwise, return the
probability of action `a` under the greedy policy. Default is None.

Returns
-------
If `a` is None:
action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``self._num2action``
If `a` is None, returns an action sampled from the distribution
over actions defined by the greedy policy.

If `a` is not None:
action_prob : float in range [0, 1]
If `a` is not None, returns the probability of `a` under the
greedy policy.
"""
        P, E = self.parameters, self.env_info
        n_actions = np.prod(E["n_actions_per_dim"])
        a_star = np.argmax([P["Q"][(s, aa)] for aa in range(n_actions)])
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def _on_policy_update(self, s, a, r, s_, a_):
        """
Update the Q function using the expected SARSA on-policy TD(0) update:

Q[s, a] <- Q[s, a] + lr * [r + temporal_discount * E[Q[s', a'] | s'] - Q[s, a]]

where

E[ Q[s', a'] | s'] is the expected value of the Q function over all
a_ given that we're in state s' under the current policy

NB. the expected SARSA update can be used for both on- and off-policy
methods. In an off-policy context, if the target policy is greedy and
the expectation is taken wrt. the target policy then the expected SARSA
update is exactly Q-learning.

Parameters
----------
s : int as returned by `self._obs2num`
The id for the state/observation at timestep t-1
a : int as returned by `self._action2num`
The id for the action taken at timestep t-1
r : float
The reward after taking action `a` in state `s` at timestep t-1
s_ : int as returned by `self._obs2num`
The id for the state/observation at timestep t
a_ : int as returned by `self._action2num`
The id for the action taken at timestep t
"""
        Q, E, pi = self.parameters["Q"], self.env_info, self.behavior_policy

        # TODO: this assumes that all actions are available in each state
        n_actions = np.prod(E["n_actions_per_dim"])

        # compute the expected value of Q(s', a') given that we are in state s'
        E_Q = np.sum([pi(s_, aa) * Q[(s_, aa)] for aa in range(n_actions)]) if s_ else 0

        # perform the expected SARSA TD(0) update
        qsa = Q[(s, a)]
        Q[(s, a)] = qsa + self.lr * (r + self.temporal_discount * E_Q - qsa)

    def _off_policy_update(self, s, a, r, s_):
        """
Update the `Q` function using the TD(0) Q-learning update:

Q[s, a] <- Q[s, a] + lr * (r + temporal_discount * max_a { Q[s', a] } - Q[s, a])

Parameters
----------
s : int as returned by `self._obs2num`
The id for the state/observation at timestep `t-1`
a : int as returned by `self._action2num`
The id for the action taken at timestep `t-1`
r : float
The reward after taking action `a` in state `s` at timestep `t-1`
s_ : int as returned by `self._obs2num`
The id for the state/observation at timestep `t`
"""
        Q, E = self.parameters["Q"], self.env_info
        n_actions = np.prod(E["n_actions_per_dim"])

        qsa = Q[(s, a)]
        Qs_ = [Q[(s_, aa)] for aa in range(n_actions)] if s_ else [0]
        Q[(s, a)] = qsa + self.lr * (r + self.temporal_discount * np.max(Qs_) - qsa)

    def update(self):
        """
Update the parameters of the model online after each new state-action.
"""
        H, HS = self.hyperparameters, self.episode_history
        (s, a), r = HS["state_actions"][-2], HS["rewards"][-1]
        s_, a_ = HS["state_actions"][-1]

        if H["off_policy"]:
            self._off_policy_update(s, a, r, s_)
        else:
            self._on_policy_update(s, a, r, s_, a_)

    def sample(self, obs):
        """
Execute the behavior policy--an :math:`\epsilon`-soft policy used to
generate actions during training.

Parameters
----------
obs : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``env.step(action)``
An observation from the environment.

Returns
-------
action : int, float, or :py:class:`ndarray <numpy.ndarray>`
An action sampled from the distribution over actions defined by the
epsilon-soft policy.
"""
        s = self._obs2num[obs]
        return self.behavior_policy(s)

    def greedy_policy(self, max_steps, render=True):
        """
Execute a deterministic greedy policy using the current agent
parameters.

Parameters
----------
max_steps : int
The maximum number of steps to run the episode.
render : bool
Whether to render the episode during execution.

Returns
-------
total_reward : float
The total reward on the episode.
n_steps : float
The total number of steps taken on the episode.
"""
        self.flush_history()

        H = self.episode_history
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            s = self._obs2num[obs]
            action = self._greedy(s)
            a = self._action2num[action]

            # store (state, action) tuple
            H["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # record rewards
            H["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps
