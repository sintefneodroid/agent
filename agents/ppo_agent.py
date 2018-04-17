#!/usr/local/bin/python
# coding: utf-8
__author__ = 'cnheider'

import cv2
import torch
import torch as th
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm

from agents.ac_agent import ACAgent

cv2.setNumThreads(0)

import utilities as U


class PPOAgent(ACAgent):
  '''
  An agent learned with PPO using Advantage Actor-Critic framework
  - Actor takes state as input
  - Critic takes both state and action as input
  - agent interact with environment to collect experience
  - agent training with experience to update policy
  - adam seems better than rmsprop for ppo
  '''

  def rollout(self, init_obs, env):
    pass

  def __init__(self, config=None, *args, **kwargs):

    self._state_dim = config.ARCH_PARAMS['input_size']
    self._action_dim = config.ARCH_PARAMS['output_size']
    self._steps = config.STEPS

    self._gamma = config.GAMMA
    self._glp = config.GAE_LAMBDA_PARAMETER
    self._horizon_penalty = config.DONE_PENALTY

    self._experience_buffer = U.ExpandableBuffer()
    self._critic_loss = config.CRITIC_LOSS
    self._actor_critic_lr = config.ACTOR_LR
    self._critic_lr = config.CRITIC_LR
    self._entropy_reg_coef = config.ENTROPY_REG_COEF
    self._value_reg_coef = config.VALUE_REG_COEF
    self._batch_size = config.BATCH_SIZE
    self._episodes_before_train = config.EPISODES_BEFORE_TRAIN
    self._target_tau = config.TARGET_TAU
    self._max_grad_norm = config.MAX_GRADIENT_NORM

    # params for epsilon greedy
    self._epsilon_start = config.EXPLORATION_EPSILON_START
    self._epsilon_end = config.EXPLORATION_EPSILON_END
    self._epsilon_decay = config.EXPLORATION_EPSILON_DECAY

    self._use_cuda = config.USE_CUDA and th.cuda.is_available()

    self._update_target_interval = config.TARGET_UPDATE_STEPS
    self._clip = config.CLIP

    self._optimiser_type = config.OPTIMISER_TYPE

    self._actor_critic_arch = U.ActorCriticNetwork
    self._actor_critic_params = config

    super().__init__(config, *args, **kwargs)

    self.__build_model__()

  def __build_model__(self):
    self._actor_critic = self._actor_critic_arch(self._actor_critic_params)
    self._actor_critic_target = U.ActorCriticNetwork(self._actor_critic_params)
    self._actor_critic_target.load_state_dict(self._actor_critic.state_dict())

    self._optimiser = self._optimiser_type(self._actor_critic.parameters(), lr=self._actor_critic_lr)

    if self._use_cuda:
      self._actor_critic.cuda()
      self._actor_critic_target.cuda()

  def maybe_take_n_steps(self, initial_state, environment, n=100):
    state = initial_state
    accumulated_signal = 0

    transitions = []
    terminated = False

    T = tqdm(range(1, n + 1), f'Step #{self._step_i}', leave=False)
    for t in T:
      self._step_i += 1
      action, value_estimates, action_prob, *_ = self.discrete_categorical_sample_model(state)

      next_state, signal, terminated, _ = environment.step(action)

      successor_state = None
      if not terminated:  # If environment terminated then there is no successor state
        successor_state = next_state

      transitions.append(
          U.ValuedTransition(state, action, action_prob, value_estimates, signal, successor_state,
                             not terminated))

      state = next_state

      accumulated_signal += signal

      if self._step_i % self._update_target_interval == 0:
        self._actor_critic_target.load_state_dict(self._actor_critic.state_dict())

      if terminated:
        break

    return transitions, accumulated_signal, terminated, state

  def trace_back_steps(self, transitions):
    n_step_summary = U.ValuedTransition(*zip(*transitions))
    signals = n_step_summary.signal
    value_estimates = n_step_summary.value_estimate

    advantages, discounted_returns = U.gae(signals,
                                           value_estimates,
                                           n_step_summary.non_terminal,
                                           self._gamma,
                                           glp=self._glp)

    i = 0
    advantage_memories = []
    for step in zip(*n_step_summary):
      step = U.ValuedTransition(*step)
      advantage_memories.append(
          U.AdvantageMemory(step.state,
                            step.action,
                            step.action_prob,
                            step.value_estimate,
                            advantages[i],
                            discounted_returns[i])
          )
      i += 1

    return advantage_memories

  def evaluate(self, batch):

    states_var = U.to_var(batch.state, use_cuda=self._use_cuda).view(-1, self._state_dim[0])
    action_var = U.to_var(batch.action, use_cuda=self._use_cuda, dtype='long')
    action_probs_var = U.to_var(batch.action_prob, use_cuda=self._use_cuda).view(-1, self._action_dim[0])
    values_var = U.to_var(batch.value_estimate, use_cuda=self._use_cuda)
    advantages_var = U.to_var(batch.advantage, use_cuda=self._use_cuda)
    returns_var = U.to_var(batch.discounted_return, use_cuda=self._use_cuda)

    action_prob = action_probs_var.gather(1, action_var)

    action_probs_t, _ = self._actor_critic_target(states_var)
    action_prob_t = action_probs_t.gather(1, action_var)

    ratio = action_prob / (action_prob_t + self._divide_by_zero_safety)

    advantage = (advantages_var - advantages_var.mean()) / (
          advantages_var.std() + self._divide_by_zero_safety)

    surrogate = ratio * advantage
    surrogate_clipped = torch.clamp(ratio, min=1. - self._clip, max=1. + self._clip) * advantage  # (L^CLIP)

    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()
    value_error = (.5 * (values_var - returns_var) ** 2.).mean()
    entropy_loss = U.entropy(action_probs_var).mean()

    cost = policy_loss + value_error * self._value_reg_coef + entropy_loss * self._entropy_reg_coef

    return cost

  def train(self):
    batch = U.AdvantageMemory(*zip(*self._experience_buffer.memory))
    cost = self.evaluate(batch)
    self.optimise_wrt(cost)

  def optimise_wrt(self, cost):
    self._optimiser.zero_grad()
    cost.backward()
    if self._max_grad_norm is not None:
      nn.utils.clip_grad_norm(self._actor_critic.parameters(), self._max_grad_norm)
    self._optimiser.step()

  def discrete_categorical_sample_model(self, state):
    state_var = U.to_var(state, use_cuda=self._use_cuda, unsqueeze=True)
    softmax_probs, value_estimate = self._actor_critic(state_var)
    m = Categorical(softmax_probs)
    action = m.sample()
    a = action.cpu().data.numpy()[0]
    return a, value_estimate, m.log_prob(action), m.probs

  def continuous_sample_model(self, state):
    state_var = U.to_var([state])
    a_mean, a_log_std, value_estimate = self._actor_critic(state_var)

    # randomly sample from normal distribution, whose mean and variance come from policy network.
    # [b, a_dim]
    a = torch.normal(a_mean, torch.exp(a_log_std))

    # value, x, states = self(inputs, states, masks)
    # action = self.dist.sample(x, deterministic=deterministic)
    # action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)

    # return value, action, action_log_probs, states

    return a

  # choose an action based on state for execution
  def sample_action(self, state):
    action, *_ = self.discrete_categorical_sample_model(state)
    return action

  def save_model(self, C):
    U.save_model(self._model, C)

  def load_model(self, model_path, evaluation=False):  # TODO: dont use _model as model
    print('Loading latest model: ' + model_path)
    self._model = self._actor_critic(**self._value_arch_parameters)
    self._model.load_state_dict(torch.load(model_path))
    if evaluation:
      self._model = self._model.eval()
      self._model.train(False)
    if self._use_cuda:
      self._model = self._model.cuda()
    else:
      self._model = self._model.cpu()


def test_ppo_agent(C):
  import gym

  U.set_seed(C.SEED)

  env = gym.make(C.ENVIRONMENT_NAME)
  env.seed(C.SEED)

  state_dim = env.observation_space.shape[0]
  if len(env.action_space.shape) >= 1:
    action_dim = env.action_space.shape[0]
  else:
    action_dim = env.action_space.n

  C.ARCH_PARAMS['input_size'] = [state_dim]
  C.ARCH_PARAMS['output_size'] = [action_dim]

  ppo_agent = PPOAgent(C)

  initial_state = env.reset()
  cs = tqdm(range(1, C.ROLLOUTS + 1), f'Rollout {0}, {0}', leave=True)
  for rollout_i in cs:

    transitions, accum_signal, terminated, initial_state = ppo_agent.maybe_take_n_steps(initial_state, env)

    if terminated:
      initial_state = env.reset()

    if rollout_i >= C.EPISODES_BEFORE_TRAIN:
      advantage_memories = ppo_agent.trace_back_steps(transitions)
      for m in advantage_memories:
        ppo_agent._experience_buffer.remember(m)

      ppo_agent.train()
      ppo_agent._experience_buffer.forget()


if __name__ == '__main__':
  import argparse
  import configs.ppo_config as C

  parser = argparse.ArgumentParser(description='PG Agent')
  parser.add_argument('--ENVIRONMENT_NAME', '-E', type=str, default=C.ENVIRONMENT_NAME,
                      metavar='ENVIRONMENT_NAME',
                      help='name of the environment to run')
  parser.add_argument('--PRETRAINED_PATH', '-T', metavar='PATH', type=str, default='',
                      help='path of pre-trained model')
  parser.add_argument('--RENDER_ENVIRONMENT', '-R', action='store_true',
                      default=C.RENDER_ENVIRONMENT,
                      help='render the environment')
  parser.add_argument('--NUM_WORKERS', '-N', type=int, default=4, metavar='NUM_WORKERS',
                      help='number of threads for agent (default: 4)')
  parser.add_argument('--SEED', '-S', type=int, default=1, metavar='SEED',
                      help='random seed (default: 1)')
  args = parser.parse_args()

  for k, arg in args.__dict__.items():
    setattr(C, k, arg)

  for k, arg in U.get_upper_vars_of(C).items():
    print(f'{k} = {arg}')

  input('\nPress any key to begin... ')

  try:
    test_ppo_agent(C)
  except KeyboardInterrupt:
    print('Stopping')
