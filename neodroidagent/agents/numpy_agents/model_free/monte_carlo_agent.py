#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

import numpy as np

from neodroidagent.agents.numpy_agents.numpy_agent import NumpyAgent


class MonteCarloAgent(NumpyAgent):
    def __init__(self, env, off_policy=False, temporal_discount=0.9, epsilon=0.1):
        """
A Monte-Carlo learning agent trained using either first-visit Monte
Carlo updates (on-policy) or incremental weighted importance sampling
(off-policy).

Parameters
----------
env : :class:`gym.wrappers` or :class:`gym.envs` instance
The environment to run the agent on.
off_policy : bool
Whether to use a behavior policy separate from the target policy
during training. If False, use the same epsilon-soft policy for
both behavior and target policies. Default is False.
temporal_discount : float between [0, 1]
The discount factor used for downweighting future rewards. Smaller
values result in greater discounting of future rewards. Default is
0.9.
epsilon : float between [0, 1]
The epsilon value in the epsilon-soft policy. Larger values
encourage greater exploration during training. Default is 0.1.
"""
        super().__init__(env)

        self.epsilon = epsilon
        self.off_policy = off_policy
        self.temporal_discount = temporal_discount

        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"
        assert not E["continuous_observations"], "Observation space must be discrete"

        n_states = np.prod(E["n_obs_per_dim"])
        n_actions = np.prod(E["n_actions_per_dim"])

        self._create_2num_dicts()

        # behavior policy is stochastic, epsilon-soft policy
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy
        if self.off_policy:
            self.parameters["C"] = np.zeros((n_states, n_actions))

            # target policy is deterministic, greedy policy
            self.target_policy = self._greedy

        # initialize Q function
        self.parameters["Q"] = np.random.rand(n_states, n_actions)

        # initialize returns object for each state-action pair
        self.derived_variables = {
            "returns": {(s, a): [] for s in range(n_states) for a in range(n_actions)},
            "episode_num": 0,
        }

        self.hyperparameters = {
            "agent": "MonteCarloAgent",
            "epsilon": self.epsilon,
            "off_policy": self.off_policy,
            "temporal_discount": self.temporal_discount,
        }

        self.episode_history = {"state_actions": [], "rewards": []}

    def _epsilon_soft_policy(self, s, a=None):
        """
Epsilon-soft exploration policy.

Notes
-----
Soft policies are necessary for first-visit Monte Carlo methods, as
they require continual exploration (i.e., each state-action pair must
have nonzero probability of occurring).

In epsilon-soft policies, :math:`\pi(a \mid s) > 0` for all :math:`s
\in S` and all :math:`a \in A(s)` at the start of training. As learning
progresses, :math:`pi` gradually shifts closer and closer to a
deterministic optimal policy.

In particular, we have:

.. math::

\pi(a \mid s)  &=  1 - \epsilon + \\frac{\epsilon}{|A(s)|}  &&\\text{if} a = a^*
\pi(a \mid s)  &=  \\frac{\epsilon}{|A(s)|}                 &&\\text{if} a \\neq a^*

where :math:`|A(s)|` is the number of actions available in state `s`
and :math:`a^* \in A(s)` is the greedy action in state `s` (i.e.,
:math:`a^* = \\arg \max_a Q(s, a)`).

Note that epsilon-greedy policies are instances of epsilon-soft
policies, defined as policies for which :math:`\pi(a|s) \geq \epsilon / |A(s)|`
for all states and actions.

Parameters
----------
s : int, float, or tuple
The state number for the current observation, as returned by
``_obs2num[obs]``.
a : int, float, tuple, or None
The action number in the current state, as returned by
``self._action2num[obs]``. If None, sample an action from the
action probabilities in state `s`, otherwise, return the
probability of action `a` under the epsilon-soft policy. Default is
None.

Returns
-------
action : int, float, or :py:class:`ndarray <numpy.ndarray>`
If `a` is None, this is an action sampled from the distribution
over actions defined by the epsilon-soft policy. If `a` is not
None, this is the probability of `a` under the epsilon-soft policy.
"""
        E, P = self.env_info, self.parameters

        # TODO: this assumes all actions are available in every state
        n_actions = np.prod(E["n_actions_per_dim"])

        a_star = P["Q"][s, :].argmax()
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
A greedy behavior policy.

Notes
-----
Only used when off-policy is True.

Parameters
----------
s : int, float, or tuple
The state number for the current observation, as returned by
``self._obs2num[obs]``.
a : int, float, or tuple
The action number in the current state, as returned by
``self._action2num[obs]``. If None, sample an action from the action
probabilities in state `s`, otherwise, return the probability of
action `a` under the greedy policy. Default is None.

Returns
-------
action : int, float, or :py:class:`ndarray <numpy.ndarray>`
If `a` is None, this is an action sampled from the distribution
over actions defined by the greedy policy. If `a` is not
None, this is the probability of `a` under the greedy policy.
"""
        a_star = self.parameters["Q"][s, :].argmax()
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def _on_policy_update(self):
        """
Update the `Q` function using an on-policy first-visit Monte Carlo
update.

Notes
-----
The on-policy first-visit Monte Carlo update is

.. math::

Q'(s, a) \leftarrow
\\text{avg}(\\text{reward following first visit to } (s, a)
\\text{ across all episodes})

RL agents seek to learn action values conditional on subsequent optimal
behavior, but they need to behave non-optimally in order to explore all
actions (to find the optimal actions).

The on-policy approach is a compromise -- it learns action values not
for the optimal policy, but for a *near*-optimal policy that still
explores (the epsilon-soft policy).
"""
        D, P, HS = self.derived_variables, self.parameters, self.episode_history

        ep_rewards = HS["rewards"]
        sa_tuples = set(HS["state_actions"])

        locs = [HS["state_actions"].index(sa) for sa in sa_tuples]
        cumulative_returns = [np.sum(ep_rewards[i:]) for i in locs]

        # update Q value with the average of the first-visit return across
        # episodes
        for (s, a), cr in zip(sa_tuples, cumulative_returns):
            D["returns"][(s, a)].append(cr)
            P["Q"][s, a] = np.mean(D["returns"][(s, a)])

    def _off_policy_update(self):
        """
Update `Q` using weighted importance sampling.

Notes
-----
In importance sampling updates, we account for the fact that we are
updating a different policy from the one we used to generate behavior
by weighting the accumulated rewards by the ratio of the probability of
the trajectory under the target policy versus its probability under
the behavior policies. This is known as the importance sampling weight.

In weighted importance sampling, we scale the accumulated rewards for a
trajectory by their importance sampling weight, then take the
*weighted* average using the importance sampling weight. This weighted
average then becomes the value for the trajectory.

W   = importance sampling weight
G_t = total discounted reward from time t until episode end
C_n = sum of importance weights for the first n rewards

This algorithm converges to Q* in the limit.
"""
        P = self.parameters
        HS = self.episode_history
        ep_rewards = HS["rewards"]
        T = len(ep_rewards)

        G, W = 0.0, 1.0
        for t in reversed(range(T)):
            s, a = HS["state_actions"][t]
            G = self.temporal_discount * G + ep_rewards[t]
            P["C"][s, a] += W

            # update Q(s, a) using weighted importance sampling
            P["Q"][s, a] += (W / P["C"][s, a]) * (G - P["Q"][s, a])

            # multiply the importance sampling ratio by the current weight
            W *= self.target_policy(s, a) / self.behavior_policy(s, a)

            if W == 0.0:
                break

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

    def run_episode(self, max_steps, render=False):
        """
Run the agent on a single episode.

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
        total_rwd, n_steps = self._episode(max_steps, render)

        D["episode_num"] += 1
        return total_rwd, n_steps

    def _episode(self, max_steps, render):
        """
Execute agent on an episode.

Parameters
----------
max_steps : int
The maximum number of steps to run the episode.
render : bool
Whether to render the episode during training.

Returns
-------
reward : float
The total reward on the episode.
steps : float
The number of steps taken on the episode.
"""
        obs = self.env.reset()
        HS = self.episode_history
        total_reward, n_steps = 0.0, 0

        for i in range(max_steps):
            if render:
                self.env.render()

            n_steps += 1
            action = self.sample(obs)

            s = self._obs2num[obs]
            a = self._action2num[action]

            # store (state, action) tuple
            HS["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)

            # record rewards
            HS["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps

    def update(self):
        """
Update the parameters of the model following the completion of an
episode. Flush the episode history after the update is complete.
"""
        H = self.hyperparameters
        if H["off_policy"]:
            self._off_policy_update()
        else:
            self._on_policy_update()

        self.flush_history()

    def greedy_policy(self, max_steps, render=True):
        """
Execute a greedy policy using the current agent parameters.

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
        H = self.episode_history
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            n_steps += 1
            action = self._greedy(obs)

            s = self._obs2num[obs]
            a = self._action2num[action]

            # store (state, action) tuple
            H["state_actions"].append((s, a))

            # take action
            obs, reward, done, info = self.env.step(action)

            # record rewards
            H["rewards"].append(reward)
            total_reward += reward

            if done:
                break

        return total_reward, n_steps
