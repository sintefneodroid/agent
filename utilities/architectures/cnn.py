#!/usr/bin/env python3
# coding=utf-8
__author__='cnheider'
from torch import nn
from torch.nn import functional as F

from utilities.initialisation.atari_weight_init import atari_initializer
from utilities.initialisation.ortho_weight_init import ortho_weights


class CNN(nn.Module):
  def __init__(self, configuration):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(configuration['input_channels'][0], 32, kernel_size=8,
                           stride=4)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
    self.fc1 = nn.Linear(7 * 7 * 64, 512)
    self.fc2 = nn.Linear(512, configuration['output_size'][0])

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(x.size(0), -1)  # Flatten
    x = F.relu(self.fc1(x))
    return self.fc2(x)


class CategoricalCNN(CNN):

  def forward(self, x):
    x = super(CategoricalCNN, self).forward(x)
    return F.softmax(x, dim=0)


class AtariCNN(nn.Module):
  def __init__(self, num_actions):
    """ Basic convolutional actor-critic network for Atari 2600 games

    Equivalent to the network in the original DQN paper.

    Args:
        num_actions (int): the number of available discrete actions
    """
    super().__init__()

    self.conv = nn.Sequential(nn.Conv2d(4, 32, 8, stride=4),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(32, 64, 4, stride=2),
                              nn.ReLU(inplace=True),
                              nn.Conv2d(64, 64, 3, stride=1),
                              nn.ReLU(inplace=True))

    self.fc = nn.Sequential(nn.Linear(64 * 7 * 7, 512),
                            nn.ReLU(inplace=True))

    self.pi = nn.Linear(512, num_actions)
    self.v = nn.Linear(512, 1)

    self.num_actions = num_actions

    # parameter initialization
    self.apply(atari_initializer)
    self.pi.weight.data = ortho_weights(self.pi.weight.size(), scale=.01)
    self.v.weight.data = ortho_weights(self.v.weight.size())

  def forward(self, conv_in):
    """ Module forward pass

    Args:
        conv_in (Variable): convolutional input, shaped [N x 4 x 84 x 84]

    Returns:
        pi (Variable): action probability logits, shaped [N x self.num_actions]
        v (Variable): value predictions, shaped [N x 1]
    """
    N = conv_in.size()[0]

    conv_out = self.conv(conv_in).view(N, 64 * 7 * 7)

    fc_out = self.fc(conv_out)

    pi_out = self.pi(fc_out)
    v_out = self.v(fc_out)

    return pi_out, v_out
