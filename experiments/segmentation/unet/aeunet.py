import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

from experiments.segmentation.unet import conv1x1
from experiments.segmentation.unet.down_convolution import DownConvolution
from experiments.segmentation.unet.up_convolution import UpConvolution


class AEUNet(nn.Module):
  """
  `UNet` class is based on https://arxiv.org/abs/1505.04597
  Contextual spatial information (from the decoding, expansive pathway) about an input tensor is merged with
  information representing the localization of details (from the encoding, compressive pathway).

  Modifications to the original paper:
  (1) padding is used in 3x3 convolutions to prevent loss of border pixels
  (2) merging outputs does not require cropping due to (1)
  (3) residual connections can be used by specifying UNet(merge_mode='add')
  (4) if non-parametric upsampling is used in the decoder
      pathway (specified by upmode='upsample'), then an
      additional 1x1 2d convolution occurs after upsampling
      to reduce channel dimensionality by a factor of 2.
      This channel halving happens with the convolution in
      the tranpose convolution (specified by upmode='transpose')
  """

  def __init__(self,
               out_channels,
               *,
               in_channels=3,
               depth=5,
               start_filters=64,
               up_mode='upsample',
               merge_mode='concat'):
    '''
    Arguments:
        in_channels: int, number of channels in the input tensor.
            Default is 3 for RGB images.
        depth: int, number of MaxPools in the U-Net.
        start_filters: int, number of convolutional filters for the            first conv.
        up_mode: string, type of upconvolution. Choices: 'transpose' for transpose convolution or 'upsample' for nearest neighbour upsampling.
        merge_mode:'concat'
    '''
    super().__init__()

    if up_mode in ('transpose', 'upsample'):
      self.up_mode = up_mode
    else:
      raise ValueError(
          f'"{up_mode}" is not a valid mode for upsampling. Only "transpose" and "upsample" are allowed.')

    if merge_mode in ('concat', 'add'):
      self.merge_mode = merge_mode
    else:
      raise ValueError(
          f'"{up_mode}" is not a valid mode for merging up and down paths. Only "concat" and "add" are '
          f'allowed.')

    if self.up_mode == 'upsample' and self.merge_mode == 'add':
      # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
      raise ValueError('up_mode "upsample" is incompatible with merge_mode "add" at the moment because '
                       'it does not make sense to use nearest neighbour to reduce depth channels (by half).')

    self.out_channels = out_channels
    self.in_channels = in_channels
    self.start_filters = start_filters
    self.depth = depth

    self.down_convolutions = []
    self.up_convolutions_ae = []
    self.up_convolutions_seg = []

    outs = None

    # create the encoder pathway and add to a list
    for i in range(depth):
      ins = self.in_channels if i == 0 else outs
      outs = self.start_filters * (2 ** i)
      pooling = True if i < depth - 1 else False

      down_conv = DownConvolution(ins, outs, pooling=pooling)
      self.down_convolutions.append(down_conv)

    outs_ae=outs
    # create the decoder pathway and add to a list - careful! decoding only requires depth-1 blocks
    for i in range(depth - 1):
      ins = outs_ae
      outs_ae = ins // 2
      up_conv = UpConvolution(ins, outs_ae, up_mode=up_mode, merge_mode=merge_mode)
      self.up_convolutions_ae.append(up_conv)

    outs_seg=outs
    # create the decoder pathway and add to a list - careful! decoding only requires depth-1 blocks
    for i in range(depth - 1):
      ins = outs_seg
      outs_seg = ins // 2
      up_conv = UpConvolution(ins, outs_seg, up_mode=up_mode, merge_mode=merge_mode)
      self.up_convolutions_seg.append(up_conv)

    self.conv_final_ae = conv1x1(outs_ae, self.in_channels)
    self.conv_final_seg = conv1x1(outs_seg, self.out_channels)

    # add the list of modules to current module
    self.down_convolutions = nn.ModuleList(self.down_convolutions)
    self.up_convolutions_ae = nn.ModuleList(self.up_convolutions_ae)
    self.up_convolutions_seg = nn.ModuleList(self.up_convolutions_seg)

    self.reset_params()

  @staticmethod
  def weight_init(m):
    if isinstance(m, nn.Conv2d):
      init.xavier_normal_(m.weight)
      init.constant_(m.bias, 0)

  def reset_params(self):
    for i, m in enumerate(self.modules()):
      self.weight_init(m)

  def forward(self, x):
    encoder_outs = []

    for i, module in enumerate(self.down_convolutions):     # encoder pathway, save outputs for merging
      x, before_pool = module(x)
      encoder_outs.append(before_pool)

    x_ae = x
    x_seg = x
    for i, module in enumerate(self.up_convolutions_seg):
      before_pool = encoder_outs[-(i + 2)]
      x_seg = module(before_pool, x_seg)

    for i, module in enumerate(self.up_convolutions_ae):
      before_pool = encoder_outs[-(i + 2)]
      x_ae = module(before_pool, x_ae)

    x_seg = self.conv_final_seg(x_seg)
    x_ae = self.conv_final_ae(x_ae)
    return x_seg, x_ae


if __name__ == "__main__":
  model = AEUNet(3, depth=2, merge_mode='concat')
  x = torch.FloatTensor(np.random.random((1, 3, 320, 320)))
  out,_ = model(x)
  loss = torch.sum(out)
  loss.backward()
  import matplotlib.pyplot as plt

  im = out.detach()
  print(im.shape)
  plt.imshow(im[0].transpose(2, 0))
  plt.show()