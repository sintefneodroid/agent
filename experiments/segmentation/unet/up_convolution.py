import torch
import torch.nn as nn

from experiments.segmentation.unet import conv3x3, upconv2x2


class UpConvolution(nn.Module):
  """
  A helper Module that performs 2 convolutions and 1 UpConvolution.
  A ReLU activation follows each convolution.
  """

  def __init__(self, in_channels, out_channels, merge_mode='concat', up_mode='transpose'):
    super().__init__()

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.merge_mode = merge_mode
    self.up_mode = up_mode

    self.upconv = upconv2x2(self.in_channels, self.out_channels, mode=self.up_mode)

    if self.merge_mode == 'concat':
      self.conv1 = conv3x3(2 * self.out_channels, self.out_channels)
    else:  # num of input channels to conv2 is same
      self.conv1 = conv3x3(self.out_channels, self.out_channels)

    self.conv2 = conv3x3(self.out_channels, self.out_channels)

  def forward(self, from_down, from_up):
    """
    Forward pass
    Arguments:
        from_down: tensor from the encoder pathway
        from_up: upconv'd tensor from the decoder pathway
    """
    from_up = self.upconv(from_up)

    if self.merge_mode == 'concat':
      x = torch.cat((from_up, from_down), 1)
    else:
      x = from_up + from_down

    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    return x
