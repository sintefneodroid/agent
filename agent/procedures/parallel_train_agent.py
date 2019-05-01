#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'cnheider'
import argparse
import os
import signal

import torch
import torch.multiprocessing as MP
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def _train(rankey, args, model):
  torch.manual_seed(args.seed + rank)

  magic_mean = 0.1307
  magic_std = 0.3081

  train_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=True, download=True,
                     transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((magic_mean,), (magic_std,))
                       ])),
      batch_size=args.batch_size, shuffle=True, num_workers=1)
  test_loader = torch.utils.data.DataLoader(
      datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((magic_mean,), (magic_std,))
        ])),
      batch_size=args.batch_size, shuffle=True, num_workers=1)

  optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
  for epoch in range(1, args.epochs + 1):
    train_epoch(epoch, args, model, train_loader, optimizer)
    test_epoch(model, test_loader)


def train_epoch(epoch, args, model, data_loader, optimizer):
  model.train()
  pid = os.getpid()
  for batch_idx, (data, target) in enumerate(data_loader):
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          pid, epoch, batch_idx * len(data), len(data_loader.dataset),
                      100. * batch_idx / len(data_loader), loss.item()))


def test_epoch(model, data_loader):
  model.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in data_loader:
      output = model(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
      pred = output.max(1)[1]  # get the index of the max log-probability
      correct += pred.eq(target).sum().item()

  test_loss /= len(data_loader.dataset)
  print(
      f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(data_loader.dataset)} '
      f'({100. * correct / len(data_loader.dataset):.0f}%)\n')


def test(rankey, args, model):
  pass


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num-processes', type=int, default=2, metavar='N',
                    help='how many training processes to use (default: 2)')


class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
    self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
    self.conv2_drop = nn.Dropout2d()
    self.fc1 = nn.Linear(320, 50)
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, 320)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = self.fc2(x)
    return F.log_softmax(x, dim=1)


if __name__ == '__main__':
  args = parser.parse_args()
  assert args.num_processes != 0, 'Num processes must not be 0'

  torch.manual_seed(args.seed)

  model = Net()
  model.share_memory()  # gradients are allocated lazily, so they are not shared here

  signal.signal(signal.SIGINT, signal.signal(signal.SIGINT, signal.SIG_IGN))
  processes = []

  p = MP.Process(target=test, args=(args.num_processes, args, model))
  p.start()
  processes.append(p)

  for rank in range(args.num_processes):
    p = MP.Process(target=_train, args=(rank, args, model))
    p.start()
    processes.append(p)

  try:
    for p in processes:
      p.join()
  except KeyboardInterrupt:
    print('Stopping training. Best model stored at {}model_best'.format(args.dump_location))
    for p in processes:
      p.terminate()
