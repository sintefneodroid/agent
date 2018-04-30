#!/usr/bin/env python3
# coding=utf-8
from itertools import count
from warnings import warn

import torch
from tqdm import tqdm

__author__ = 'cnheider'

from abc import ABC, abstractmethod

import utilities as U


class Agent(ABC):
  '''
All agent should inherit from this class
'''

  def __init__(self, config=None, *args, **kwargs):
    self._step_i = 0
    self._rollout_i = 0
    self._end_training = False
    self._input_size = None
    self._output_size = None
    self._divide_by_zero_safety = 1e-10
    self._use_cuda = False
    self._device = torch.device(
        'cuda:0' if torch.cuda.is_available() and self._use_cuda else 'cpu'
        )

    self._verbose = False

    self.__local_defaults__()

    if config:
      self.set_config_attributes(config)

  def build_agent(self, env, device, **kwargs):
    self._infer_input_output_sizes(env)
    self._device = device

    self.__build_models__()

  @abstractmethod
  def __build_models__(self):
    raise NotImplementedError

  @abstractmethod
  def __local_defaults__(self):
    raise NotImplementedError

  def stop_training(self):
    self._end_training = True

  @abstractmethod
  def sample_action(self, state, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def __sample_model__(self, state, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def __optimise_wrt__(self, error, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def evaluate(self, batch, *args, **kwargs):
    raise NotImplementedError()

  @abstractmethod
  def rollout(self, init_obs, env, *args, **kwargs):
    raise NotImplementedError()

  def _infer_input_output_sizes(self, env, *args, **kwargs):
    '''
Tries to infer input and output size from env if either _input_size or _output_size, is None or -1 (int)

:rtype: object
'''
    if self._input_size is None or self._input_size == -1:
      self._input_size = env.observation_space.shape
    print('observation dimensions: ', self._input_size)

    if self._output_size is None or self._output_size == -1:
      if hasattr(env.action_space, 'num_binary_actions'):
        self._output_size = [env.action_space.num_binary_actions]
      elif len(env.action_space.shape) >= 1:
        self._output_size = env.action_space.shape
      else:
        self._output_size = [env.action_space.n]
    print('action dimensions: ', self._output_size)

  def set_config_attributes(self, config, *args, **kwargs):
    if config:
      config_vars = U.get_upper_vars_of(config)
      self._check_for_duplicates_in_args(**config_vars)
      self._parse_set_attr(**config_vars)
    self._parse_set_attr(**kwargs)

  def _check_for_duplicates_in_args(self, *args, **kwargs):
    for k, v in kwargs.items():

      occur = 0

      if kwargs.get(k) is not None:
        occur += 1
      else:
        pass

      if k.isupper():
        k_lowered = f'_{k.lower()}'
        if kwargs.get(k_lowered) is not None:
          occur += 1
        else:
          pass
      else:
        k_lowered = f'{k.lstrip("_").upper()}'
        if kwargs.get(k_lowered) is not None:
          occur += 1
        else:
          pass

      if occur > 1:
        warn(
            f'Config contains hiding duplicates of {k} and {k_lowered}, {occur} times'
            )

  def _parse_set_attr(self, *args, **kwargs):
    for k, v in kwargs.items():
      if k.isupper():
        k_lowered = f'_{k.lower()}'
        self.__setattr__(k_lowered, v)
      else:
        self.__setattr__(k, v)

  def run(self, environment, render=True, *args, **kwargs):
    E = count(1)
    E = tqdm(E, leave=False)
    for episode_i in E:
      print('Episode {}'.format(episode_i))

      state = environment.reset()
      F = count(1)
      F = tqdm(F, leave=False)
      for frame_i in F:

        action = self.__sample_model__(state)
        state, reward, terminated, info = environment.step(action)
        if render:
          environment.render()

        if terminated:
          break

#
# def parallel():
#   args = parser.parse_args()
#   torch.manual_seed(args.seed)
#
#   if not os.path.exists(args.dump_location):
#     os.makedirs(args.dump_location)
#
#   logging.basicConfig(
#       filename=args.dump_location +
#                'train.log',
#       level=logging.INFO)
#
#   assert args.evaluate == 0 or args.num_processes == 0, \
#     'Can't train while evaluating, either n=0 or e=0'
#
#   class Net(torch.nn.Module):
#     def __init__(self, args):
#       super(Net, self).__init__()
#       self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
#       self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
#       self.conv2_drop = torch.nn.Dropout2d()
#       self.fc1 = torch.nn.Linear(320, 50)
#       self.fc2 = torch.nn.Linear(50, 10)
#
#     def forward(self, x):
#       x = F.relu(F.max_pool2d(self.conv1(x), 2))
#       x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#       x = x.view(-1, 320)
#       x = F.relu(self.fc1(x))
#       x = F.dropout(x, training=self.training)
#       x = self.fc2(x)
#       return F.log_softmax(x, dim=1)
#
#   def train(rank, args, model):
#     torch.manual_seed(args.seed + rank)
#
#     pass
#
#   def test(rank, args, model):
#     torch.manual_seed(args.seed + rank)
#
#     pass
#
#   shared_model = Net(args)
#
#   if args.load != '0':
#     shared_model.load_state_dict(torch.load(args.load))
#   shared_model.share_memory()
#
#   signal.signal(signal.SIGINT, signal.signal(signal.SIGINT, signal.SIG_IGN))
#   processes = []
#
#   p = TMP.Process(target=test, args=(args.num_processes, args, shared_model))
#   p.start()
#   processes.append(p)
#
#   for rank in range(0, args.num_processes):
#     p = TMP.Process(target=train, args=(rank, args, shared_model))
#     p.start()
#     processes.append(p)
#
#   try:
#     for p in processes:
#       p.join()
#   except KeyboardInterrupt:
#     print('Stopping training. ' +
#           'Best model stored at {}model_best'.format(args.dump_location))
#     for p in processes:
#       p.terminate()
