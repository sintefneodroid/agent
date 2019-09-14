#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any

from draugr.writers import MockWriter, Writer
from neodroidagent.agents.model_free.actor_critic.actor_critic_agent import ActorCriticAgent
from trolls.wrappers import NormalisedActions

__author__ = 'Christian Heider Nielsen'
__doc__ = r'''

           Created on 9/5/19
           '''


class SACAgent(ActorCriticAgent):
  '''
  Soft Actor Critic
  '''

  def _sample_model(self, state, **kwargs) -> Any:
    pass

  def _update(self, *args, metric_writer: Writer = MockWriter(), **kwargs) -> None:
    pass


if __name__ == '__main__':

  import random
  import os

  import numpy as np

  import torch
  import torch.nn as nn
  import torch.optim as optim
  import torch.nn.functional as F
  from torch.distributions import Normal

  from IPython.display import clear_output
  import matplotlib.pyplot as plt
  from matplotlib import animation
  from IPython.display import display
  import gym

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  env_name = "LunarLanderContinuous-v2"

  # env_name = "Pendulum-v0"
  def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title(f'episode {frame_idx}. reward: {rewards[-1]}')
    plt.plot(rewards)
    plt.show()


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


  def update(batch_size, gamma=0.99, soft_tau=1e-2, ):

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state = torch.FloatTensor(state).to(device)
    next_state = torch.FloatTensor(next_state).to(device)
    action = torch.FloatTensor(action).to(device)
    reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
    done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

    predicted_q_value1 = soft_q_net1(state, action)
    predicted_q_value2 = soft_q_net2(state, action)
    predicted_value = value_net(state)
    new_action, log_prob, epsilon, mean, log_std = policy_net.evaluate(state)

    # Training Q Function
    target_value = target_value_net(next_state)
    # we update the two Q function param by reducing the MSE (minimum squared error) between the predicted
    # Q value for a state-action pair and its corresponding target_q_value
    target_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss1 = soft_q_criterion1(predicted_q_value1, target_q_value.detach())
    q_value_loss2 = soft_q_criterion2(predicted_q_value2, target_q_value.detach())
    # print("Q Loss")
    # print(q_value_loss1)
    # clears gradient
    soft_q_optimizer1.zero_grad()
    # passaggio di backward
    q_value_loss1.backward()
    # optimization step
    soft_q_optimizer1.step()
    soft_q_optimizer2.zero_grad()
    q_value_loss2.backward()
    soft_q_optimizer2.step()
    # Training Value Function
    # for the V network update we take the minimun of the two Q values
    predicted_new_q_value = torch.min(soft_q_net1(state, new_action), soft_q_net2(state, new_action))
    # substract from it the policy's log probability of selecting that action in that state
    target_value_func = predicted_new_q_value - log_prob
    # we decrese the MSE between the above quantity and the predicted V value of that state
    value_loss = value_criterion(predicted_value, target_value_func.detach())
    # print("V Loss")
    # print(value_loss)
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()
    # Training Policy Function
    # we update the policy by reducing the policy's log probability of choosing an action in a state log(
    # pi(s)) - predicted Q-Value of that state-action pair

    policy_loss = (log_prob - predicted_new_q_value).mean()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    # Here we use the Polyak for the target value network
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
      target_param.data.copy_(
        target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


  class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
      super(ValueNetwork, self).__init__()

      self.linear1 = nn.Linear(state_dim, hidden_dim)
      self.linear2 = nn.Linear(hidden_dim, hidden_dim)
      self.linear3 = nn.Linear(hidden_dim, 1)

      self.linear3.weight.data.uniform_(-init_w, init_w)
      self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
      x = F.relu(self.linear1(state))
      x = F.relu(self.linear2(x))
      x = self.linear3(x)
      return x


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


  # The policy network has two outputs: The mean and the log standard deviation

  class PolicyNetwork(nn.Module):
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
      # log is clamped to be iin a san region.
      log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

      return mean, log_std

    def evaluate(self, state, epsilon=1e-6):
      mean, log_std = self.forward(state)
      std = log_std.exp()

      normal = Normal(0, 1)
      z = normal.sample()
      action = torch.tanh(mean + std * z.to(device))
      log_prob = Normal(mean, std).log_prob(mean + std * z.to(device)) - torch.log(
        1 - action.pow(2) + epsilon)
      return action, log_prob, z, mean, log_std

    def get_action(self, state):
      # Then to get the action we use the reparameterization trick
      state = torch.FloatTensor(state).unsqueeze(0).to(device)
      mean, log_std = self.forward(state)
      std = log_std.exp()

      # we sample a noise from a Standard Normal distribution
      normal = Normal(0, 1)
      # multiply it with our standard devation
      z = normal.sample().to(device)
      # add it to the mean and make it activated with a tanh to give our function
      action = torch.tanh(mean + std * z)

      action = action.cpu()
      return action[0]


  env = NormalisedActions(gym.make(env_name))

  # store information about action and state dimensions
  action_dim = env.action_space.shape[0]
  state_dim = env.observation_space.shape[0]

  # we are setting the hyperparameter of how many hidden layers
  # we want in our network
  hidden_dim = 256

  # inizialize the tree networks that we want to train along with a target V network
  # NB: we have two Q networks to solve the problem of overestimation of Q-Values, we use the minimun of
  # the two networks to do out policy and V function updates
  value_net = ValueNetwork(state_dim, hidden_dim).to(device)
  target_value_net = ValueNetwork(state_dim, hidden_dim).to(device)

  soft_q_net1 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
  soft_q_net2 = SoftQNetwork(state_dim, action_dim, hidden_dim).to(device)
  policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

  # The target for the Q training depends on the V network and the target for the V training depends on the
  # Q training. This makes training very unstable
  # The solution is to use a set of parameters which come close to the parameters of the main V network but
  # with a time delay
  # we use Polyak averaging
  for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

  # we initialize the main and the target V network to have the same parameters

  value_criterion = nn.MSELoss()
  soft_q_criterion1 = nn.MSELoss()
  soft_q_criterion2 = nn.MSELoss()

  value_lr = 3e-4
  soft_q_lr = 3e-4
  policy_lr = 3e-4

  value_optimizer = optim.Adam(value_net.parameters(), lr=value_lr)
  soft_q_optimizer1 = optim.Adam(soft_q_net1.parameters(), lr=soft_q_lr)
  soft_q_optimizer2 = optim.Adam(soft_q_net2.parameters(), lr=soft_q_lr)
  policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)

  replay_buffer_size = 1000000
  replay_buffer = ReplayBuffer(replay_buffer_size)

  n_episodes = 5000
  max_frames = 500000
  max_steps = 500
  frame_idx = 0
  rewards = []
  batch_size = 128
  start_episode = 0

  # The outer loop initializes the environment for the beginning of the episode
  # Until we have enough observations in the buffer

  policy_net, policy_optimizer, frame_idx, rewards = load_checkpoint(policy_net, policy_optimizer,
                                                                     'resumeTrain500000fr')

  print(f"Resuming from {frame_idx}")


  def train():
    global frame_idx, start_episode
    while frame_idx < max_frames:
      state = env.reset()
      episode_reward = 0

      # The inner loop is for the individual steps within an episode
      # here we sample an action from the policy network or random
      # and record state,action,reward,next state and done in the replay buffer
      for step in range(max_steps):
        if frame_idx > 1500:
          action = policy_net.get_action(state).detach()
          next_state, reward, done, _ = env.step(action.numpy())
        else:
          action = env.action_space.sample()
          next_state, reward, done, _ = env.step(action)

        # if start_episode%10 == 0:
        #     env.render(mode = 'rgb_array')

        replay_buffer.push(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward
        frame_idx += 1

        # we do network updates in each run of the inner loop after recording the buffer
        if len(replay_buffer) > batch_size:
          update(batch_size)

        if done:
          break
      state = {'epoch':     frame_idx + 1,
               'state_dict':policy_net.state_dict(),
               'optimizer': policy_optimizer.state_dict(),
               'rewards':   rewards
               }
      torch.save(state, 'resumeTrain500000fr')
      start_episode += 1

      print("\r frame {} reward: {}".format(frame_idx, episode_reward))
      rewards.append(episode_reward)
    plot(frame_idx, rewards)
    torch.save(policy_net, 'Train500000fr')


  # train()

  def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    # plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(start_episode):
      patch.set_data(frames[start_episode])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    display(anim)


  # Run a demo of the environment
  state = env.reset()
  cum_reward = 0
  frames = []
  for t in range(50000):
    # Render into buffer.
    # env.render(mode = 'rgb_array')
    frames.append(env.render(mode='rgb_array'))
    action = policy_net.get_action(state)
    state, reward, done, info = env.step(action.detach().numpy())
    if done:
      break
  env.close()
  display_frames_as_gif(frames)
