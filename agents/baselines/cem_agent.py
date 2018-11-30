import time
from itertools import count
from typing import Any, Tuple

import draugr
import gym
import numpy as np
from gym.spaces import Box, Discrete
from tqdm import tqdm

import utilities as U
from agents.abstract.dfo_agent import DFOAgent


class CEMAgent(DFOAgent):
  '''
  Cross Entropy Method (CEM)

  The idea is to initialize the mean and sigma of a Gaussian and then for n_iter times we:

    1. collect batch_size samples of theta from a Gaussian with the current mean and sigma
    2. perform a noisy evaluation to get the total rewards with these thetas
    3. select n_elite of the best thetas into an elite set
    4. upate our mean and sigma to be that from the elite set

  CartPole env
  ____________________________________________________

           │               ┌───theta ~ N(mean,std)───┐
           │
     4 observations        [[ 2.2  4.5 ]
  [-0.1 -0.4  0.06  0.5] *  [ 3.4  0.2 ]  + [[ 0.2 ]
           |                [ 4.2  3.4 ]     [ 1.1 ]]
           │                [ 0.1  9.0 ]]
           |                     W              b
      ┌────o────┐
  <─0─│2 actions│─1─>    = [-0.4  0.1] ──argmax()─> 1
      └─o─────o─┘
  ____________________________________________________

  '''

  # region Private

  def __defaults__(self) -> None:
    self._policy = None

  # endregion

  # region Protected

  def _build(self, **kwargs) -> None:
    pass

  class DeterministicDiscreteActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
      """
      dim_ob: dimension of observations
      n_actions: number of actions
      theta: flat vector of parameters
      """
      dim_ob = ob_space.shape[0]
      n_actions = ac_space.n
      assert len(theta) == (dim_ob + 1) * n_actions
      self.W = theta[0: dim_ob * n_actions].reshape(dim_ob, n_actions)
      self.b = theta[dim_ob * n_actions: None].reshape(1, n_actions)

    def act(self, ob):
      """
      """
      y = ob.dot(self.W) + self.b
      a = y.argmax()
      return a

  class DeterministicContinuousActionLinearPolicy(object):

    def __init__(self, theta, ob_space, ac_space):
      """
      dim_ob: dimension of observations
      dim_ac: dimension of action vector
      theta: flat vector of parameters
      """
      self.ac_space = ac_space
      dim_ob = ob_space.shape[0]
      dim_ac = ac_space.shape[0]
      assert len(theta) == (dim_ob + 1) * dim_ac
      self.W = theta[0: dim_ob * dim_ac].reshape(dim_ob, dim_ac)
      self.b = theta[dim_ob * dim_ac: None]

    def act(self, ob):
      a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
      return a

  def rolloutm(self, policy, env, num_steps, discount=1.0, render=False):
    disc_total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
      a = policy.act(ob)
      (ob, signal, terminal, _info) = env.step(a)
      disc_total_rew += signal * discount ** t
      if render and t % 3 == 0:
        env.render()
      if terminal:
        break
    return disc_total_rew

  def noisy_evaluation(self, env, num_steps, theta, discount=0.90):
    policy = self.make_policy(env, theta)
    signal = self.rolloutm(policy,
                           env,
                           num_steps,
                           discount)
    return signal

  def make_policy(self, env, theta):
    if isinstance(env.action_space, Discrete):
      return self.DeterministicDiscreteActionLinearPolicy(theta,
                                                          env.observation_space,
                                                          env.action_space)
    elif isinstance(env.action_space, Box):
      return self.DeterministicContinuousActionLinearPolicy(theta,
                                                            env.observation_space,
                                                            env.action_space)
    else:
      raise NotImplementedError

  def _optimise_wrt(self, error, *args, **kwargs) -> None:
    pass

  def _sample_model(self, state, *args, **kwargs) -> Any:
    pass

  def _train(self,
             _environment,
             rollouts=2000,
             render=False,
             render_frequency=100,
             stat_frequency=10,
             **kwargs) -> Tuple[Any, Any]:
    training_start_timestamp = time.time()
    E = range(1, rollouts)
    E = tqdm(E, f'Episode: {1}', leave=False)

    stats = draugr.StatisticCollection(stats=('signal', 'duration'))

    for episode_i in E:
      initial_state = _environment.reset()

      if episode_i % stat_frequency == 0:
        draugr.terminal_plot_stats_shared_x(
            stats,
            printer=E.write,
            )

        E.set_description(f'Epi: {episode_i}, Dur: {stats.duration.running_value[-1]:.1f}')

      if render and episode_i % render_frequency == 0:
        signal, dur, *extras = self.rollout(
            initial_state, _environment, render=render
            )
      else:
        signal, dur, *extras = self.rollout(initial_state, _environment)

      stats.duration.append(dur)
      stats.signal.append(signal)

      if self._end_training:
        break

    time_elapsed = time.time() - training_start_timestamp
    end_message = f'Training done, time elapsed: {time_elapsed // 60:.0f}m {time_elapsed %60:.0f}s'
    print('\n{} {} {}\n'.format('-' * 9, end_message, '-' * 9))

    return self._policy, stats

  # endregion

  # region Public

  def sample_action(self, state, *args, **kwargs) -> Any:
    return self._environment.action_space.sample()

  def update(self, *args, **kwargs) -> None:
    pass

  def evaluate(self, batch, *args, **kwargs) -> Any:
    pass

  def rollout(self, initial_state, environment, *, train=True, render=False, **kwargs) -> Any:
    if train:
      self._rollout_i += 1

    episode_signal = 0
    episode_length = 0

    state = initial_state

    T = count(1)
    T = tqdm(T, f'Rollout #{self._rollout_i}', leave=False)

    for t in T:
      action = self.sample_action(state)

      state, signal, terminated, info = environment.step(action=action)
      episode_signal += signal

      if render:
        environment.render()

      if terminated:
        episode_length = t
        break

    if train:
      self.update()

    return episode_signal, episode_length

  def load(self, *args, **kwargs) -> None:
    pass

  def save(self, *args, **kwargs) -> None:
    pass

  # endregion

  def t(self):
    # Task settings:
    env = gym.make('CartPole-v0')  # Change as needed
    num_steps = 500  # maximum length of episode

    # Alg settings:
    n_iter = 2000  # number of iterations of CEM
    batch_size = 25  # number of samples per batch
    elite_frac = 0.2  # fraction of samples used as elite set
    n_elite = int(batch_size * elite_frac)
    extra_std = 2.0
    extra_decay_time = 10

    if isinstance(env.action_space, Discrete):
      n = env.action_space.n
    elif isinstance(env.action_space, Box):
      n = env.action_space.shape[0]
    else:
      raise NotImplementedError

    dim_theta = (env.observation_space.shape[0] + 1) * n

    # Initialize mean and standard deviation
    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)

    # Now, for the algorithm
    for itr in range(n_iter):
      # Sample parameter vectors
      extra_cov = max(1.0 - itr / extra_decay_time, 0) * extra_std ** 2
      thetas = np.random.multivariate_normal(mean=theta_mean,
                                             cov=np.diag(np.array(theta_std ** 2) + extra_cov),
                                             size=batch_size)
      rewards = np.array(map(self.noisy_evaluation, thetas))

      # Get elite parameters
      elite_inds = rewards.argsort()[-n_elite:]
      elite_thetas = thetas[elite_inds]

      # Update theta_mean, theta_std
      theta_mean = elite_thetas.mean(axis=0)
      theta_std = elite_thetas.std(axis=0)
      # print(f'iteration {itr:d}. mean f: {np.mean(rewards):8.3g}. max f: {np.max(rewards):8.3g}')
      self.rolloutm(self.make_policy(env,
                                     theta_mean),
                    env,
                    num_steps,
                    discount=0.90,
                    render=True)

    env.close()


if __name__ == '__main__':
  import configs.agent_test_configs.test_pg_config as C

  C.CONNECT_TO_RUNNING = False
  C.ENVIRONMENT_NAME = 'grd'

  # test_agent_main(CEMAgent, C)
  CEMAgent().t()
