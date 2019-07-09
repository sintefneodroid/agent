#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

# LSTM Memory
from agent.interfaces.architecture import Architecture

LSTM_MEMORY = 128


class LSTM_DQN(Architecture):

  def __init__(self, n_action):
    super().__init__()
    self._n_action = n_action

    self.conv1 = nn.Conv2d(
        4, 32, kernel_size=8, stride=1, padding=1
        )  # (In Channel, Out Channel, ...)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
    self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    self.lstm = nn.LSTM(16, LSTM_MEMORY, 1)  # (Input, Hidden, Num Layers)

    self.affine1 = nn.Linear(LSTM_MEMORY * 64, 512)
    # self.affine2 = nn.Linear(2048, 512)
    self.affine2 = nn.Linear(512, self._n_action)

  def forward(self, x, hidden_state, cell_state):
    # CNN
    h = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
    h = F.relu(F.max_pool2d(self.conv2(h), kernel_size=2, stride=2))
    h = F.relu(F.max_pool2d(self.conv3(h), kernel_size=2, stride=2))
    h = F.relu(F.max_pool2d(self.conv4(h), kernel_size=2, stride=2))

    # LSTM
    h = h.view(h.size(0), h.size(1), 16)  # (32, 64, 4, 4) -> (32, 64, 16)
    h, (next_hidden_state, next_cell_state) = self.lstm(
        h, (hidden_state, cell_state)
        )
    h = h.view(h.size(0), -1)  # (32, 64, 256) -> (32, 16348)

    # Fully Connected Layers
    h = F.relu(self.affine1(h.view(h.size(0), -1)))
    # h = F.relu(self.affine2(h.view(h.size(0), -1)))
    h = self.affine2(h)
    return h, next_hidden_state, next_cell_state

  def init_states(self) -> [Variable, Variable]:
    hidden_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
    cell_state = Variable(torch.zeros(1, 64, LSTM_MEMORY).cuda())
    return hidden_state, cell_state

  def reset_states(self, hidden_state, cell_state):
    hidden_state[:, :, :] = 0
    cell_state[:, :, :] = 0
    return hidden_state.detach(), cell_state.detach()
