#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = "Christian Heider Nielsen"
__doc__ = r"""

           Created on 27/02/2020
           """

import numpy as np

from neodroidagent.agents.numpy_agents.numpy_agent import NumpyAgent


class CrossEntropyAgent(NumpyAgent):
    def __init__(self, env, n_samples_per_episode=500, retain_percentage=0.2):
        """
A cross-entropy method agent.

Notes
-----
The cross-entropy method agent only operates on ``envs`` with discrete
action spaces.

On each episode the agent generates `n_theta_samples` of the parameters
(:math:`\\theta`) for its behavior policy. The `i`'th sample at
timestep `t` is:

.. math::

\\theta_i  &=  \{\mathbf{W}_i^{(t)}, \mathbf{b}_i^{(t)} \} \\\\
\\theta_i  &\sim  \mathcal{N}(\mu^{(t)}, \Sigma^{(t)})

Weights (:math:`\mathbf{W}_i`) and bias (:math:`\mathbf{b}_i`) are the
parameters of the softmax policy:

.. math::

\mathbf{z}_i  &=  \\text{obs} \cdot \mathbf{W}_i + \mathbf{b}_i \\\\
p(a_i^{(t + 1)})  &=  \\frac{e^{\mathbf{z}_i}}{\sum_j e^{z_{ij}}} \\\\
a^{(t + 1)}  &=  \\arg \max_j p(a_j^{(t+1)})

At the end of each episode, the agent takes the top `retain_prcnt`
highest scoring :math:`\\theta` samples and combines them to generate
the mean and variance of the distribution of :math:`\\theta` for the
next episode:

.. math::

\mu^{(t+1)}  &=  \\text{avg}(\\texttt{best_thetas}^{(t)}) \\\\
\Sigma^{(t+1)}  &=  \\text{var}(\\texttt{best_thetas}^{(t)})

Parameters
----------
env : :meth:`gym.wrappers` or :meth:`gym.envs` instance
The environment to run the agent on.
n_samples_per_episode : int
The number of theta samples to evaluate on each episode. Default is 500.
retain_percentage: float
The percentage of `n_samples_per_episode` to use when calculating
the parameter update at the end of the episode. Default is 0.2.
"""
        super().__init__(env)

        self.retain_prcnt = retain_percentage
        self.n_samples_per_episode = n_samples_per_episode
        self._init_params()

    def _init_params(self):
        E = self.env_info
        assert not E["continuous_actions"], "Action space must be discrete"

        self._create_2num_dicts()
        b_len = np.prod(E["n_actions_per_dim"])
        W_len = b_len * np.prod(E["obs_dim"])
        theta_dim = b_len + W_len

        # init mean and variance for mv gaussian with dimensions theta_dim
        theta_mean = np.random.rand(theta_dim)
        theta_var = np.ones(theta_dim)

        self.parameters = {"theta_mean": theta_mean, "theta_var": theta_var}
        self.derived_variables = {
            "b_len": b_len,
            "W_len": W_len,
            "W_samples": [],
            "b_samples": [],
            "episode_num": 0,
            "cumulative_rewards": [],
        }

        self.hyperparameters = {
            "agent": "CrossEntropyAgent",
            "retain_prcnt": self.retain_prcnt,
            "n_samples_per_episode": self.n_samples_per_episode,
        }

        self.episode_history = {"rewards": [], "state_actions": []}

    def sample(self, obs):
        """
Generate actions according to a softmax policy.

Notes
-----
The softmax policy assumes that the pmf over actions in state :math:`x_t` is
given by:

.. math::

\pi(a | x^{(t)}) = \\text{softmax}(
\\text{obs}^{(t)} \cdot \mathbf{W}_i^{(t)} + \mathbf{b}_i^{(t)} )

where :math:`\mathbf{W}` is a learned weight matrix, `obs` is the observation
at timestep `t`, and **b** is a learned bias vector.

Parameters
----------
obs : int or :py:class:`ndarray <numpy.ndarray>`
An observation from the environment.

Returns
-------
action : int, float, or :py:class:`ndarray <numpy.ndarray>`
An action sampled from the distribution over actions defined by the
softmax policy.
"""
        E, P = self.env_info, self.parameters
        W, b = P["W"], P["b"]

        s = self._obs2num[obs]
        s = np.array([s]) if E["obs_dim"] == 1 else s

        # compute softmax
        Z = s.T @ W + b
        e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        action_probs = e_Z / e_Z.sum(axis=-1, keepdims=True)

        # sample action
        a = np.random.multinomial(1, action_probs).argmax()
        return self._num2action[a]

    def run_episode(self, max_steps, render=False):
        """
Run the agent on a single episode.

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
        self._sample_thetas()

        E, D = self.env_info, self.derived_variables
        n_actions = np.prod(E["n_actions_per_dim"])
        W_len, obs_dim = D["W_len"], E["obs_dim"]
        steps, rewards = [], []

        for theta in D["theta_samples"]:
            W = theta[:W_len].reshape(obs_dim, n_actions)
            b = theta[W_len:]

            total_rwd, n_steps = self._episode(W, b, max_steps, render)
            rewards.append(total_rwd)
            steps.append(n_steps)

        # return the average reward and average number of steps across all
        # samples on the current episode
        D["episode_num"] += 1
        D["cumulative_rewards"] = rewards
        return np.mean(D["cumulative_rewards"]), np.mean(steps)

    def _episode(self, W, b, max_steps, render):
        """
Run the agent for an episode.

Parameters
----------
W : :py:class:`ndarray <numpy.ndarray>` of shape `(obs_dim, n_actions)`
The weights for the softmax policy.
b : :py:class:`ndarray <numpy.ndarray>` of shape `(bias_len, )`
The bias for the softmax policy.
max_steps : int
The maximum number of steps to run the episode.
render : bool
Whether to render the episode during training.

Returns
-------
reward : float
The total reward on the episode.
steps : float
The total number of steps taken on the episode.
"""
        rwds, sa = [], []
        H = self.episode_history
        total_reward, n_steps = 0.0, 1
        obs = self.env.reset()

        self.parameters["W"] = W
        self.parameters["b"] = b

        for i in range(max_steps):
            if render:
                self.env.render()

            n_steps += 1
            action = self.sample(obs)
            s, a = self._obs2num[obs], self._action2num[action]
            sa.append((s, a))

            obs, reward, done, _ = self.env.step(action)
            rwds.append(reward)
            total_reward += reward

            if done:
                break

        H["rewards"].append(rwds)
        H["state_actions"].append(sa)
        return total_reward, n_steps

    def update(self):
        """
Update :math:`\mu` and :math:`\Sigma` according to the rewards accrued on
the current episode.

Returns
-------
avg_reward : float
The average reward earned by the best `retain_prcnt` theta samples
on the current episode.
"""
        D, P = self.derived_variables, self.parameters
        n_retain = int(self.retain_prcnt * self.n_samples_per_episode)

        # sort the cumulative rewards for each theta sample from greatest to least
        sorted_y_val_idxs = np.argsort(D["cumulative_rewards"])[::-1]
        top_idxs = sorted_y_val_idxs[:n_retain]

        # update theta_mean and theta_var with the best theta value
        P["theta_mean"] = np.mean(D["theta_samples"][top_idxs], axis=0)
        P["theta_var"] = np.var(D["theta_samples"][top_idxs], axis=0)

    def _sample_thetas(self):
        """
Sample `n_samples_per_episode` thetas from a multivariate Gaussian with
mean `theta_mean` and covariance `diag(theta_var)`
"""
        P, N = self.parameters, self.n_samples_per_episode
        Mu, Sigma = P["theta_mean"], np.diag(P["theta_var"])
        samples = np.random.multivariate_normal(Mu, Sigma, N)
        self.derived_variables["theta_samples"] = samples

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
        E, D, P = self.env_info, self.derived_variables, self.parameters
        Mu, Sigma = P["theta_mean"], np.diag(P["theta_var"])
        sample = np.random.multivariate_normal(Mu, Sigma, 1)

        W_len, obs_dim = D["W_len"], E["obs_dim"]
        n_actions = np.prod(E["n_actions_per_dim"])

        W = sample[0, :W_len].reshape(obs_dim, n_actions)
        b = sample[0, W_len:]
        total_reward, n_steps = self._episode(W, b, max_steps, render)
        return total_reward, n_steps
