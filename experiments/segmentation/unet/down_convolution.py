import torch
import torch.nn as nn

from experiments.segmentation.unet import conv3x3


class DownConvolution(nn.Module):
  """
  A helper Module that performs 2 convolutions and 1 MaxPool.
  A ReLU activation follows each convolution.
  """

  def __init__(self, in_channels, out_channels, pooling=True):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.pooling = pooling

    self.conv1 = conv3x3(self.in_channels, self.out_channels)
    self.conv2 = conv3x3(self.out_channels, self.out_channels)

    if self.pooling:
      self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self, x):
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    before_pool = x

    if self.pooling:
      x = self.pool(x)

    return x, before_pool
