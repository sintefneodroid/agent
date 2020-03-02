from collections import defaultdict
from typing import Iterable

import numpy as np

from neodroidagent.agents.numpy_agents.numpy_agent import NumpyAgent
from neodroidagent.utilities.misc import tile_state_space, EnvModel


class DynaAgent(NumpyAgent):
    def __init__(
        self,
        env,
        lr: float = 0.4,
        epsilon: float = 0.1,
        n_tilings: int = 8,
        obs_max=None,
        obs_min=None,
        q_plus=False,
        grid_dims: Iterable = [8, 8],
        explore_weight: float = 0.05,
        temporal_discount: float = 0.9,
        n_simulated_actions: int = 50,
    ):
        """
A Dyna-`Q` / Dyna-`Q+` agent with full TD(0) `Q`-learning updates via
prioritized-sweeping.

Parameters
----------
env : :class:`gym.wrappers` or :class:`gym.envs` instance
The environment to run the agent on
lr : float
Learning rate for the `Q` function updates. Default is 0.05.
epsilon : float between [0, 1]
The epsilon value in the epsilon-soft policy. Larger values
encourage greater exploration during training. Default is 0.1.
n_tilings : int
The number of overlapping tilings to use if the env observation
space is continuous. Unused if observation space is discrete.
Default is 8.
obs_max : float or :py:class:`ndarray <numpy.ndarray>` or None
The value to treat as the max value of the observation space when
calculating the grid widths if the observation space is continuous.
If None, use :meth:`env.observation_space.high`. Unused if observation
space is discrete. Default is None.
obs_min : float or :py:class:`ndarray <numpy.ndarray>` or None
The value to treat as the min value of the observation space when
calculating grid widths if the observation space is continuous. If
None, use :meth:`env.observation_space.low`. Unused if observation
space is discrete. Default is None.
grid_dims : list
The number of rows and columns in each tiling grid if the env
observation space is continuous. Unused if observation space is
discrete. Default is `[8, 8]`.
q_plus : bool
Whether to add incentives for visiting states that the agent hasn't
encountered recently. Default is False.
explore_weight : float
Amount to incentivize exploring states that the agent hasn't
recently visited. Only used if `q_plus` is True. Default is 0.05.
temporal_discount : float between [0, 1]
The discount factor used for downweighting future rewards. Smaller
values result in greater discounting of future rewards. Default is
0.9.
n_simulated_actions : int
THe number of simulated actions to perform for each "real" action.
Default is 50.
"""
        super().__init__(env)

        self.lr = lr
        self.q_plus = q_plus
        self.obs_max = obs_max
        self.obs_min = obs_min
        self.epsilon = epsilon
        self.n_tilings = n_tilings
        self.grid_dims = grid_dims
        self.explore_weight = explore_weight
        self.temporal_discount = temporal_discount
        self.n_simulated_actions = n_simulated_actions

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
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy

        # initialize Q function and model
        self.parameters["Q"] = defaultdict(np.random.rand)
        self.parameters["model"] = EnvModel()

        # initialize returns object for each state-action pair
        self.derived_variables = {
            "episode_num": 0,
            "sweep_queue": {},
            "visited": set([]),
            "steps_since_last_visit": defaultdict(lambda: 0),
        }

        if self.q_plus:
            self.derived_variables["steps_since_last_visit"] = defaultdict(
                np.random.rand
            )

        self.hyperparameters = {
            "agent": "DynaAgent",
            "lr": self.lr,
            "q_plus": self.q_plus,
            "obs_max": self.obs_max,
            "obs_min": self.obs_min,
            "epsilon": self.epsilon,
            "n_tilings": self.n_tilings,
            "grid_dims": self.grid_dims,
            "explore_weight": self.explore_weight,
            "temporal_discount": self.temporal_discount,
            "n_simulated_actions": self.n_simulated_actions,
        }

        self.episode_history = {"state_actions": [], "rewards": []}

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

    def _epsilon_soft_policy(self, s, a=None):
        """
Epsilon-soft exploration policy.

In epsilon-soft policies, pi(a|s) > 0 for all s ∈ S and all a ∈ A(s) at
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
self._obs2num[obs]
a : int, float, or tuple
The action number in the current state, as returned by
self._action2num[obs]. If None, sample an action from the action
probabilities in state s, otherwise, return the probability of
action `a` under the epsilon-soft policy. Default is None.

Returns
-------
If `a` is None:
action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by :meth:`_num2action`
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
A greedy behavior policy.

Parameters
----------
s : int, float, or tuple
The state number for the current observation, as returned by
self._obs2num[obs]
a : int, float, or tuple
The action number in the current state, as returned by
self._action2num[obs]. If None, sample an action from the action
probabilities in state s, otherwise, return the probability of
action `a` under the greedy policy. Default is None.

Returns
-------
If `a` is None:
action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by :meth:`_num2action`
If `a` is None, returns an action sampled from the distribution
over actions defined by the greedy policy.

If `a` is not None:
action_prob : float in range [0, 1]
If `a` is not None, returns the probability of `a` under the
greedy policy.
"""
        E, Q = self.env_info, self.parameters["Q"]
        n_actions = np.prod(E["n_actions_per_dim"])
        a_star = np.argmax([Q[(s, aa)] for aa in range(n_actions)])
        if a is None:
            out = self._num2action[a_star]
        else:
            out = 1 if a == a_star else 0
        return out

    def update(self):
        """
Update the priority queue with the most recent (state, action) pair and
perform random-sample one-step tabular Q-planning.

Notes
-----
The planning algorithm uses a priority queue to retrieve the
state-action pairs from the agent's history which will result in the
largest change to its `Q`-value if backed up. When the first pair in
the queue is backed up, the effect on each of its predecessor pairs is
computed. If the predecessor's priority is greater than a small
threshold the pair is added to the queue and the process is repeated
until either the queue is empty or we exceed `n_simulated_actions`
updates.
"""
        s, a = self.episode_history["state_actions"][-1]
        self._update_queue(s, a)
        self._simulate_behavior()

    def _update_queue(self, s, a):
        """
Update the priority queue by calculating the priority for (s, a) and
inserting it into the queue if it exceeds a fixed (small) threshold.

Parameters
----------
s : int as returned by `self._obs2num`
The id for the state/observation
a : int as returned by `self._action2num`
The id for the action taken from state `s`
"""
        sweep_queue = self.derived_variables["sweep_queue"]

        # TODO: what's a good threshold here?
        priority = self._calc_priority(s, a)
        if priority >= 0.001:
            if (s, a) in sweep_queue:
                sweep_queue[(s, a)] = max(priority, sweep_queue[(s, a)])
            else:
                sweep_queue[(s, a)] = priority

    def _calc_priority(self, s, a):
        """
Compute the "priority" for state-action pair (s, a). The priority P is
defined as:

P = sum_{s_} p(s_) * abs(r + temporal_discount * max_a {Q[s_, a]} - Q[s, a])

which corresponds to the absolute magnitude of the TD(0) Q-learning
backup for (s, a).

Parameters
----------
s : int as returned by `self._obs2num`
The id for the state/observation
a : int as returned by `self._action2num`
The id for the action taken from state `s`

Returns
-------
priority : float
The absolute magnitude of the full-backup TD(0) Q-learning update
for (s, a)
"""
        priority = 0.0
        E = self.env_info
        Q = self.parameters["Q"]
        env_model = self.parameters["model"]
        n_actions = np.prod(E["n_actions_per_dim"])

        outcome_probs = env_model.outcome_probs(s, a)
        for (r, s_), p_rs_ in outcome_probs:
            max_q = np.max([Q[(s_, aa)] for aa in range(n_actions)])
            P = p_rs_ * (r + self.temporal_discount * max_q - Q[(s, a)])
            priority += np.abs(P)
        return priority

    def _simulate_behavior(self):
        """
Perform random-sample one-step tabular Q-planning with prioritized
sweeping.

Notes
-----
This approach uses a priority queue to retrieve the state-action pairs
from the agent's history with largest change to their Q-values if
backed up. When the first pair in the queue is backed up, the effect on
each of its predecessor pairs is computed. If the predecessor's
priority is greater than a small threshold the pair is added to the
queue and the process is repeated until either the queue is empty or we
have exceeded a `n_simulated_actions` updates.
"""
        env_model = self.parameters["model"]
        sweep_queue = self.derived_variables["sweep_queue"]
        for _ in range(self.n_simulated_actions):
            if len(sweep_queue) == 0:
                break

            # select (s, a) pair with the largest update (priority)
            sq_items = list(sweep_queue.items())
            (s_sim, a_sim), _ = sorted(sq_items, key=lambda x: x[1], reverse=True)[0]

            # remove entry from queue
            del sweep_queue[(s_sim, a_sim)]

            # update Q function for (s_sim, a_sim) using the full-backup
            # version of the TD(0) Q-learning update
            self._update(s_sim, a_sim)

            # get all (_s, _a) pairs that lead to s_sim (ie., s_sim's predecessors)
            pairs = env_model.state_action_pairs_leading_to_outcome(s_sim)

            # add predecessors to queue if their priority exceeds thresh
            for (_s, _a) in pairs:
                self._update_queue(_s, _a)

    def _update(self, s, a):
        """
Update Q using a full-backup version of the TD(0) Q-learning update:

Q(s, a) = Q(s, a) + lr *
sum_{r, s'} [
p(r, s' | s, a) * (r + gamma * max_a { Q(s', a) } - Q(s, a))
]

Parameters
----------
s : int as returned by ``self._obs2num``
The id for the state/observation
a : int as returned by ``self._action2num``
The id for the action taken from state `s`
"""
        update = 0.0
        env_model = self.parameters["model"]
        E, D, Q = self.env_info, self.derived_variables, self.parameters["Q"]
        n_actions = np.prod(E["n_actions_per_dim"])

        # sample rewards from the model
        outcome_probs = env_model.outcome_probs(s, a)
        for (r, s_), p_rs_ in outcome_probs:
            # encourage visiting long-untried actions by adding a "bonus"
            # reward proportional to the sqrt of the time since last visit
            if self.q_plus:
                r += self.explore_weight * np.sqrt(D["steps_since_last_visit"][(s, a)])

            max_q = np.max([Q[(s_, a_)] for a_ in range(n_actions)])
            update += p_rs_ * (r + self.temporal_discount * max_q - Q[(s, a)])

        # update Q value for (s, a) pair
        Q[(s, a)] += self.lr * update

    def run_episode(self, max_steps, render=False):
        """
Run the agent on a single episode without performing `Q`-function
backups.

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
Whether to perform the `Q` function backups after each step.
Default is True.

Returns
-------
reward : float
The total reward on the episode.
steps : float
The number of steps taken on the episode.
"""
        self.flush_history()

        obs = self.env.reset()
        env_model = self.parameters["model"]
        HS, D = self.episode_history, self.derived_variables

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

            # update model
            env_model[(s, a, reward, s_)] += 1

            # update history counter
            for k in D["steps_since_last_visit"].keys():
                D["steps_since_last_visit"][k] += 1
            D["steps_since_last_visit"][(s, a)] = 0

            if update:
                self.update()

            # store next (state, action) tuple
            HS["state_actions"].append((s_, a_))
            s, a = s_, a_

            if done:
                break

        return total_reward, n_steps

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
