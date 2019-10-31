#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
from typing import Any, Dict, Sequence, Tuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output, display
from matplotlib import animation, pyplot
from torch.distributions import Normal

from draugr.torch_utilities.initialisation.seeding import get_torch_device
from draugr.writers import MockWriter, Writer
from neodroid.utilities import ActionSpace, EnvironmentSnapshot, ObservationSpace, SignalSpace
from neodroidagent import PROJECT_APP_PATH
from neodroidagent.agents.torch_agents.model_free.actor_critic import ActorCriticAgent
from neodroidagent.architectures import Architecture
from trolls.wrappers import NormalisedActions

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 9/5/19
           '''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env_name = "LunarLanderContinuous-v2"
SAC_STATE_PATH = PROJECT_APP_PATH.user_data / 'sac_state.pth'
soft_tau = 1e-2


def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
  # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
  start_epoch = 0
  rewards = []
  if os.path.isfile(filename):
    print(f"=> loading checkpoint '{filename}'")
    checkpoint = torch.load(filename)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    rewards = checkpoint['rewards']
    print(f"=> loaded checkpoint '{filename}' (epoch {checkpoint['epoch']})")
  else:
    print(f"=> no checkpoint found at '{filename}'")

  return model, optimizer, start_epoch, rewards


class ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = capacity
    self.buffer = []
    self.position = 0

  def push(self, state, action, reward, next_state, done):
    if len(self.buffer) < self.capacity:
      self.buffer.append(None)
    self.buffer[self.position] = (state, action, reward, next_state, done)
    self.position = (self.position + 1) % self.capacity

  def sample(self, batch_size):
    batch = random.sample(self.buffer, batch_size)
    state, action, reward, next_state, done = map(np.stack, zip(*batch))
    return state, action, reward, next_state, done

  def __len__(self):
    return len(self.buffer)


class SoftQNetwork(nn.Module):
  def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
    super(SoftQNetwork, self).__init__()

    self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)
    self.linear3 = nn.Linear(hidden_size, 1)

    self.linear3.weight.data.uniform_(-init_w, init_w)
    self.linear3.bias.data.uniform_(-init_w, init_w)

  def forward(self, state, action):
    x = torch.cat([state, action], 1)
    x = F.relu(self.linear1(x))
    x = F.relu(self.linear2(x))
    x = self.linear3(x)
    return x


class PolicyNetwork(nn.Module):
  '''
  # The policy network has two outputs: The mean and the log standard deviation
  '''

  def __init__(self,
               num_inputs,
               num_actions,
               hidden_size,
               init_w=3e-3,
               log_std_min=-20,
               log_std_max=2):
    super(PolicyNetwork, self).__init__()

    self.log_std_min = log_std_min
    self.log_std_max = log_std_max

    self.linear1 = nn.Linear(num_inputs, hidden_size)
    self.linear2 = nn.Linear(hidden_size, hidden_size)

    self.mean_linear = nn.Linear(hidden_size, num_actions)
    self.mean_linear.weight.data.uniform_(-init_w, init_w)
    self.mean_linear.bias.data.uniform_(-init_w, init_w)

    self.log_std_linear = nn.Linear(hidden_size, num_actions)
    self.log_std_linear.weight.data.uniform_(-init_w, init_w)
    self.log_std_linear.bias.data.uniform_(-init_w, init_w)

  def forward(self, state):
    x = F.relu(self.linear1(state))
    x = F.relu(self.linear2(x))

    mean = self.mean_linear(x)
    log_std = self.log_std_linear(x)
    log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

    return mean, log_std

  def evaluate(self, state, epsilon=1e-6):
    mean, log_std = self.forward(state)
    std = log_std.exp()

    normal = Normal(0, 1)
    z = normal.sample()
    action = torch.tanh(mean + std * z.to(get_torch_device()))
    log_prob = Normal(mean, std).log_prob(mean + std * z.to(get_torch_device())) - torch.log(
      1 - action.pow(2) + epsilon)
    return action, log_prob, z, mean, log_std

  def get_action(self, state):
    # Then to get the action we use the reparameterization trick
    state = torch.FloatTensor(state).unsqueeze(0).to(get_torch_device())
    mean, log_std = self.forward(state)
    std = log_std.exp()

    # we sample a noise from a Standard Normal distribution
    normal = Normal(0, 1)
    # multiply it with our standard devation
    z = normal.sample().to(get_torch_device())
    # add it to the mean and make it activated with a tanh to give our function
    action = torch.tanh(mean + std * z)

    action = action.cpu()
    return action[0]


class SACAgent(ActorCriticAgent):
  '''
  Soft Actor Critic
  '''

  @property
  def models(self) -> Dict[str, Architecture]:
    return {'critic1':self.q1, 'critic2':self.soft_q_net1, 'actor':self.policy_net}

  def _sample(self, state: EnvironmentSnapshot, *args, no_random: bool = False,
              metric_writer: Writer = MockWriter(), **kwargs) -> Tuple[Sequence, Any]:
    return self._sample_model(state)

  def __build__(self,
                observation_space: ObservationSpace,
                action_space: ActionSpace,
                signal_space: SignalSpace,
                **kwargs) -> None:
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]

    hidden_dim = 256

    self.q1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(get_torch_device())
    self.q1_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(get_torch_device())
    self._update_target(target_model=self.q1_target, source_model=self.q1, copy_percentage=1)

    self.q2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(get_torch_device())
    self.q2_target = SoftQNetwork(state_dim, action_dim, hidden_dim).to(get_torch_device())
    self._update_target(target_model=self.q2_target, source_model=self.q2, copy_percentage=1)

    self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(get_torch_device())

    self.q_criterion = nn.MSELoss()

    soft_q_lr = 3e-4
    policy_lr = 3e-4

    self.soft_q_optimizer1 = optim.Adam(self.q1.parameters(), lr=soft_q_lr)
    self.soft_q_optimizer2 = optim.Adam(self.q2.parameters(), lr=soft_q_lr)
    self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

    replay_buffer_size = 1000000
    self._memory = ReplayBuffer(replay_buffer_size)

    # The outer loop initializes the environment for the beginning of the episode
    # Until we have enough observations in the buffer

    self.policy_net, self.policy_optimizer, frame_idx, rewards = load_checkpoint(self.policy_net,
                                                                                 self.policy_optimizer,
                                                                                 SAC_STATE_PATH)

    print(f"Resuming from {frame_idx}")

  def _sample_model(self, state, **kwargs) -> Any:
    with torch.no_grad():
      return agent.policy_net.get_action(state).detach().cpu().numpy()

  def _update(self,
              batch_size,
              *args,
              gamma: float = 0.99,
              metric_writer: Writer = MockWriter(),
              **kwargs) -> None:
    state, action, reward, next_state, terminals = self._memory.sample(batch_size)

    state = torch.FloatTensor(state).to(get_torch_device())
    next_state = torch.FloatTensor(next_state).to(get_torch_device())
    action = torch.FloatTensor(action).to(get_torch_device())
    reward = torch.FloatTensor(reward).unsqueeze(1).to(get_torch_device())
    terminals = torch.FloatTensor(np.float32(terminals)).unsqueeze(1).to(get_torch_device())

    predicted_q_value1 = self.q1(state, action)
    predicted_q_value2 = self.q2(state, action)
    new_action, log_prob, epsilon, mean, log_std = self.policy_net.evaluate(state)

    q1_target_value = self.q1_target(next_state, new_action)
    q1_target_q_value = reward + (1 - terminals) * gamma * q1_target_value
    q_value_loss1 = self.q_criterion(predicted_q_value1, q1_target_q_value.detach())

    q2_target_value = self.q1_target(next_state, new_action)
    q2_target_q_value = reward + (1 - terminals) * gamma * q2_target_value
    q_value_loss2 = self.q_criterion(predicted_q_value2, q2_target_q_value.detach())

    self.soft_q_optimizer1.zero_grad()
    q_value_loss1.backward()
    self.soft_q_optimizer1.step()

    self.soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    self.soft_q_optimizer2.step()

    predicted_new_q_value = torch.min(q1_target_value,
                                      q2_target_value)

    policy_loss = (log_prob - predicted_new_q_value).mean()

    self.policy_optimizer.zero_grad()
    policy_loss.backward()
    self.policy_optimizer.step()

  def update_targets(self):
    self._update_target(target_model=self.q1_target,
                        source_model=self.q1,
                        copy_percentage=soft_tau)

    self._update_target(target_model=self.q2_target,
                        source_model=self.q2,
                        copy_percentage=soft_tau)


if __name__ == '__main__':

  agent = SACAgent()

  # env_name = "Pendulum-v0"
  env = NormalisedActions(gym.make(env_name))

  agent.build(env.observation_space, env.action_space, env.action_space)


  def train(max_frames=500000,
            max_steps=500,
            batch_size=128,
            start_episode=0):

    frame_idx = 0
    rewards = []

    def plot(frame_idx, rewards):
      clear_output(True)
      pyplot.figure(figsize=(20, 5))
      pyplot.subplot(131)
      pyplot.title(f'episode {frame_idx}. reward: {rewards[-1]}')
      pyplot.plot(rewards)
      pyplot.show()

    while frame_idx < max_frames:
      state = env.reset()
      episode_reward = 0

      for step in range(max_steps):
        if frame_idx > 1500:
          agent.sample(state)
          next_state, reward, done, _ = env.step(action)
        else:
          action = env.action_space.sample()
          next_state, reward, done, _ = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        if len(agent._memory) > batch_size:
          agent.update(batch_size)  # network updates in each run of the inner loop after recording the buffer

        if done:
          break

      state = {'epoch':     frame_idx + 1,
               'state_dict':agent.policy_net.state_dict(),
               'optimizer': agent.policy_optimizer.state_dict(),
               'rewards':   rewards
               }
      torch.save(state, SAC_STATE_PATH)
      start_episode += 1

      print(f"\r frame {frame_idx} reward: {episode_reward}")
      rewards.append(episode_reward)
    plot(frame_idx, rewards)
    torch.save(agent.policy_net, SAC_STATE_PATH)


  def inference():

    def display_frames_as_gif(frames):
      """
      Displays a list of frames as a gif, with controls
      """
      # pyplot.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
      patch = pyplot.imshow(frames[0])
      pyplot.axis('off')

      def animate(start_episode):
        patch.set_data(frames[start_episode])

      anim = animation.FuncAnimation(pyplot.gcf(), animate, frames=len(frames), interval=50)
      display(anim)

    # Run a demo of the environment
    state = env.reset()

    frames = []
    for t in range(50000):
      frames.append(env.render(mode='rgb_array'))
      action = agent.policy_net.get_action(state)
      state, reward, done, info = env.step(action.detach().numpy())
      if done:
        break
    env.close()
    display_frames_as_gif(frames)


  train()
  # inference()
