from itertools import count
from typing import Any

import numpy
from tqdm import tqdm

from draugr import Writer, MockWriter
from draugr.drawers.drawer import Drawer, MockDrawer
from neodroid.environments import VectorUnityEnvironment, EnvironmentSnapshot
from neodroid.environments.environment import Environment
from neodroidagent.agents.agent import Agent
from neodroidagent.agents.torch_agents.model_free.on_policy.policy_agent import PolicyAgent


def run(self,
        environment: VectorUnityEnvironment,
        render: bool = True) -> None:
  state = environment.reset().observables

  F = count(1)
  F = tqdm(F, leave=False, disable=not render)
  for frame_i in F:
    F.set_description(f'Frame {frame_i}')

    action, *_ = self.sample(state, no_random=True)
    state, signal, terminated, info = environment.react(action, render=render)

    if terminated.all():
      state = environment.reset().observables


def rollout_pg(agent: PolicyAgent,
               initial_state: EnvironmentSnapshot,
               environment: Environment,
               *,
               render: bool = False,
               metric_writer: Writer = MockWriter(),
               rollout_drawer: Drawer = MockDrawer(),
               train: bool = True,
               max_length: int = None,
               disable_stdout: bool = False,
               **kwargs):
  '''Perform a single rollout until termination in environment

  :param rollout_drawer:
  :param disable_stdout:
  :param metric_writer:
  :type max_length: int
  :param max_length:
  :type train: bool
  :type render: bool
  :param initial_state: The initial state observation in the environment
  :param environment: The environment the agent interacts with
  :param render: Whether to render environment interaction
  :param train: Whether the agent should use the rollout to update its model
  :param kwargs:
  :return:
    -episode_signal (:py:class:`float`) - first output
    -episode_length-
    -average_episode_entropy-
  '''

  episode_signal = []
  episode_length = 0
  episode_entropy = []

  state = initial_state.observables

  '''
  with draugr.scroll_plot_class(self._distribution_regressor.output_shape,
                                render=render,
                                window_length=66) as s:
                                '''
  for t in tqdm(count(1), f'Update #{agent._update_i}', leave=False, disable=disable_stdout):
    action, action_log_prob, entropy = agent.sample(state)

    snapshot = environment.react(action)

    state, signal, terminated = snapshot.observables, snapshot.signal, snapshot.terminated

    if agent._signal_clipping:
      signal = numpy.clip(signal,
                          agent._signal_clip_low,
                          agent._signal_clip_high)

    episode_signal.append(signal)
    episode_entropy.append(entropy.to('cpu').numpy())
    if train:
      agent._trajectory_trace.add_point(signal, action_log_prob, entropy)

    if render:
      environment.render()
      # s.draw(to_one_hot(self._distribution_regressor.output_shape, action)[0])


    if numpy.array(terminated).all() or (max_length and t > max_length):
      episode_length = t
      break

  if train:
    agent.update()

  ep = numpy.array(episode_signal).sum(axis=0).mean()
  el = episode_length
  ee = numpy.array(episode_entropy).mean(axis=0).mean()

  if metric_writer:
    metric_writer.scalar('duration', el, agent._update_i)
    metric_writer.scalar('signal', ep, agent._update_i)
    metric_writer.scalar('entropy', ee, agent._update_i)

  return ep, el, ee


def infer(self, env, render=True):
  for episode_i in count(1):
    print('Episode {}'.format(episode_i))
    state = env.reset()

    for frame_i in count(1):

      action, *_ = self.sample(state)
      state, signal, terminated, info = env.act(action)
      if render:
        env.render()

      if terminated:
        break


def rollout(self,
            initial_state: EnvironmentSnapshot,
            environment: VectorUnityEnvironment,
            *,
            train: bool = True,
            render: bool = False,
            **kwargs) -> Any:
  self._update_i += 1

  state = initial_state.observables
  episode_signal = []
  episode_length = []

  T = count(1)
  T = tqdm(T, f'Rollout #', leave=False, disable=not render)

  for t in T:
    self._sample_i += 1

    action = self.sample(state)
    snapshot = environment.react(action)

    (next_state, signal, terminated) = (snapshot.observables,
                                        snapshot.signal,
                                        snapshot.terminated)

    episode_signal.append(signal)

    if terminated.all():
      episode_length = t
      break

    state = next_state

  ep = numpy.array(episode_signal).sum(axis=0).mean()
  el = episode_length

  return ep, el


def take_n_steps(self,
                 initial_state: EnvironmentSnapshot,
                 environment: VectorUnityEnvironment,
                 n: int = 100,
                 *,
                 train: bool = False,
                 render: bool = False,
                 **kwargs) -> Any:
  state = initial_state.observables

  accumulated_signal = []

  snapshot = None
  transitions = []
  terminated = False
  T = tqdm(range(1, n + 1),
           f'Step #{self._sample_i} - {0}/{n}',
           leave=False,
           disable=not render)
  for _ in T:
    self._sample_i += 1
    action, *_ = self.sample(state)

    snapshot = environment.react(action)

    (successor_state, signal, terminated) = (snapshot.observables,
                                             snapshot.signal,
                                             snapshot.terminated)

    transitions.append((state, successor_state, signal, terminated))

    state = successor_state

    accumulated_signal += signal

    if numpy.array(terminated).all():
      snapshot = environment.reset()
      (state, signal, terminated) = (snapshot.observables,
                                     snapshot.signal,
                                     snapshot.terminated)

  return transitions, accumulated_signal, terminated, snapshot
