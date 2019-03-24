#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import enum
from collections import namedtuple
from functools import wraps
from typing import Sized

import gym

__author__ = 'cnheider'

from multiprocessing import Pipe, Process

import numpy as np


class EnvironmentWorkerCommands(enum.Enum):
  step = enum.auto()
  reset = enum.auto()
  close = enum.auto()
  get_spaces = enum.auto()
  render = enum.auto()
  seed = enum.auto()


EWC = EnvironmentWorkerCommands

EnvironmentCommand = namedtuple('EnvironmentCommand', ('command', 'data'))

EC = EnvironmentCommand

GymTuple = namedtuple('GymTuple',('observation', 'signal', 'terminal', 'info'))

def make_env(env_nam):
  @wraps(env_nam)
  def wrapper():
    env = gym.make(env_nam)
    return env

  return wrapper

def environment_worker(remote, parent_remote, env_fn_wrapper, auto_reset_on_terminal=False):
  parent_remote.close()
  env = env_fn_wrapper.x()
  terminated = False
  while True:
    cmd, data = remote.recv()
    if cmd is EWC.step:
      observation, signal, terminal, info = env.step(data)
      if terminated:
        signal=0
      if terminal:
        terminated=True
        if auto_reset_on_terminal:
          observation = env.reset()
          terminated=False
      remote.send(GymTuple(observation, signal, terminal, info))
    elif cmd is EWC.reset:
      observation = env.reset()
      terminated = False
      remote.send(observation)
    elif cmd is EWC.close:
      remote.close()
      break
    elif cmd is EWC.get_spaces:
      remote.send((env.observation_space, env.action_space))
    elif cmd is EWC.render:
      env.render()
    elif cmd is EWC.seed:
      env.seed(data)
    else:
      raise NotImplementedError


class MultipleEnvironments(object):
  """
  An abstract asynchronous, vectorized environment.
  """

  def __init__(self, num_envs, observation_space, action_space):
    self._num_envs = num_envs
    self._observation_space = observation_space
    self._action_space = action_space

  @property
  def observation_space(self):
    return self._observation_space

  @property
  def action_space(self):
    return self._action_space

  def reset(self):
    '''
    Reset all the environment_utilities and return an array of
    observations, or a tuple of observation arrays.
    If step_async is still doing work, that work will
    be cancelled and step_wait() should not be called
    until step_async() is invoked again.
    '''
    raise NotImplementedError

  def step_async(self, actions):
    '''
    Tell all the environment_utilities to start taking a step
    with the given actions.
    Call step_wait() to get the results of the step.
    You should not call this if a step_async run is
    already pending.
    '''
    raise NotImplementedError

  def step_wait(self):
    '''
    Wait for the step taken with step_async().
    Returns (obs, signals, terminals, infos):
     - obs: an array of observations, or a tuple of
            arrays of observations.
     - signals: an array of rewards
     - terminals: an array of "episode terminal" booleans
     - infos: a sequence of info objects
    '''
    raise NotImplementedError

  def close(self):
    '''
    Clean up the environment_utilities' resources.
    '''
    raise NotImplementedError

  def step(self, actions):
    self.step_async(actions)
    return self.step_wait()


class CloudpickleWrapper(object):
  """
  Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
  """

  def __init__(self, x):
    self.x = x

  def __getstate__(self):
    import cloudpickle
    return cloudpickle.dumps(self.x)

  def __setstate__(self, ob):
    import pickle
    self.x = pickle.loads(ob)


class SubProcessEnvironments(MultipleEnvironments):
  def __init__(self, environments):
    """
    envs: list of gym environment_utilities to run in subprocesses
    """
    self._waiting = False
    self._closed = False
    self._num_envs = len(environments)
    self._remotes, self._work_remotes = zip(*[Pipe() for _ in range(self._num_envs)])
    self._processes = [Process(target=environment_worker,
                               args=(work_remote,
                                     remote,
                                     CloudpickleWrapper(env))
                               )
                       for (work_remote, remote, env) in zip(self._work_remotes,
                                                             self._remotes,
                                                             environments)]
    for p in self._processes:
      p.daemon = True  # if the main process crashes, we should not cause things to hang
      p.start()
    for remote in self._work_remotes:
      remote.close()

    self._remotes[0].send(EC(EWC.get_spaces, None))
    observation_space, action_space = self._remotes[0].recv()
    super().__init__(len(environments), observation_space, action_space)

  def seed(self, seed):
    if isinstance(seed, Sized):
      assert len(seed) == self._num_envs
      for remote, s in zip(self._remotes, seed):
        remote.send(EC(EWC.seed, s))
    else:
      for remote in self._remotes:
        remote.send(EC(EWC.seed, seed))

  def render(self):
    for remote in self._remotes:
      remote.send((EWC.render, None))

  def step_async(self, actions):
    for remote, action in zip(self._remotes, actions):
      remote.send(EC(EWC.step, action))
    self._waiting = True

  def step_wait(self):
    results = [remote.recv() for remote in self._remotes]
    self._waiting = False
    obs, signals, terminals, infos = zip(*results)
    return np.stack(obs), np.stack(signals), np.stack(terminals), infos

  def reset(self):
    for remote in self._remotes:
      remote.send(EC(EWC.reset, None))
    return np.stack([remote.recv() for remote in self._remotes])

  def close(self):
    if self._closed:
      return
    if self._waiting:
      for remote in self._remotes:
        remote.recv()
    for remote in self._remotes:
      remote.send(EC(EWC.close, None))
    for p in self._processes:
      p.join()
      self._closed = True

  def __len__(self):
    return self._num_envs
