import torch.nn as nn

from experiments.segmentation.models.utils import conv2DBatchNormRelu


class SegNetDown2(nn.Module):
  def __init__(self, in_size, out_size):
    super(SegNetDown2, self).__init__()
    self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
    self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
    self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    unpooled_shape = outputs.size()
    outputs, indices = self.maxpool_with_argmax(outputs)
    return outputs, indices, unpooled_shape


class SegNetDown3(nn.Module):
  def __init__(self, in_size, out_size):
    super(SegNetDown3, self).__init__()
    self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
    self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
    self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
    self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

  def forward(self, inputs):
    outputs = self.conv1(inputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    unpooled_shape = outputs.size()
    outputs, indices = self.maxpool_with_argmax(outputs)
    return outputs, indices, unpooled_shape


class SegNetUp2(nn.Module):
  def __init__(self, in_size, out_size):
    super(SegNetUp2, self).__init__()
    self.unpool = nn.MaxUnpool2d(2, 2)
    self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
    self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

  def forward(self, inputs, indices, output_shape):
    outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
    outputs = self.conv1(outputs)
    outputs = self.conv2(outputs)
    return outputs


class SegNetUp3(nn.Module):
  def __init__(self, in_size, out_size):
    super(SegNetUp3, self).__init__()
    self.unpool = nn.MaxUnpool2d(2, 2)
    self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
    self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
    self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

  def forward(self, inputs, indices, output_shape):
    outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
    outputs = self.conv1(outputs)
    outputs = self.conv2(outputs)
    outputs = self.conv3(outputs)
    return outputs


class SegNetArch(nn.Module):
  def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
    super(SegNetArch, self).__init__()

    self.in_channels = in_channels
    self.is_unpooling = is_unpooling

    self.down1 = SegNetDown2(self.in_channels, 64)
    self.down2 = SegNetDown2(64, 128)
    self.down3 = SegNetDown3(128, 256)
    self.down4 = SegNetDown3(256, 512)
    self.down5 = SegNetDown3(512, 512)

    self.up5 = SegNetUp3(512, 512)
    self.up4 = SegNetUp3(512, 256)
    self.up3 = SegNetUp3(256, 128)
    self.up2 = SegNetUp2(128, 64)
    self.up1 = SegNetUp2(64, n_classes)

  def forward(self, inputs):

    down1, indices_1, unpool_shape1 = self.down1(inputs)
    down2, indices_2, unpool_shape2 = self.down2(down1)
    down3, indices_3, unpool_shape3 = self.down3(down2)
    down4, indices_4, unpool_shape4 = self.down4(down3)
    down5, indices_5, unpool_shape5 = self.down5(down4)

    up5 = self.up5(down5, indices_5, unpool_shape5)
    up4 = self.up4(up5, indices_4, unpool_shape4)
    up3 = self.up3(up4, indices_3, unpool_shape3)
    up2 = self.up2(up3, indices_2, unpool_shape2)
    up1 = self.up1(up2, indices_1, unpool_shape1)

    return up1

  def init_vgg16_params(self, vgg16):
    blocks = [self.down1, self.down2, self.down3, self.down4, self.down5]

    features = list(vgg16.features.children())

    vgg_layers = []
    for _layer in features:
      if isinstance(_layer, nn.Conv2d):
        vgg_layers.append(_layer)

    merged_layers = []
    for idx, conv_block in enumerate(blocks):
      if idx < 2:
        units = [conv_block.conv1.cbr_unit, conv_block.conv2.cbr_unit]
      else:
        units = [
          conv_block.conv1.cbr_unit,
          conv_block.conv2.cbr_unit,
          conv_block.conv3.cbr_unit,
          ]
      for _unit in units:
        for _layer in _unit:
          if isinstance(_layer, nn.Conv2d):
            merged_layers.append(_layer)

    assert len(vgg_layers) == len(merged_layers)

    for l1, l2 in zip(vgg_layers, merged_layers):
      if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
        assert l1.weight.size() == l2.weight.size()
        assert l1.bias.size() == l2.bias.size()
        l2.weight.data = l1.weight.data
        l2.bias.data = l1.bias.data
