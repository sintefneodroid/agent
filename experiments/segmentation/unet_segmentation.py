#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import time
from collections import defaultdict

import torchvision.transforms as T
from PIL import Image
from neodroid.wrappers.observation_wrapper.observation_wrapper import ObservationWrapper

from experiments.segmentation import helper
from experiments.segmentation.loss import calc_loss
from experiments.segmentation.unet import UNet

__author__ = 'cnheider'

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

tqdm.monitor_interval = 0

resize = T.Compose([
  # T.ToPILImage(),
  # T.Resize(40, interpolation=Image.CUBIC),
  T.ToTensor()])


def train_model(model, optimizer, scheduler, batch_size=6, num_updates=25000, device='cpu'):
  def data_loader(device):
    while True:
      a = []
      b = []
      while len(a) < batch_size:
        env.observe()
        obs = env.observer('MaterialIdSegmentationCameraObserver').observation_value
        # obs2 = env.observer("MaterialIdSegmentationSegmentationObserver").observation_value
        # blue plastic: RGBA(0.078, 0.000, 0.851, 1.000)
        # material0: RGBA(0.651, 0.004, 0.349, 1.000)
        obs1 = env.observer('RGBCameraObserver').observation_value
        a_a = np.array(Image.open(obs1).convert("RGB"))
        a.append(transform(a_a))
        b_a = np.array(Image.open(obs).convert("RGB"))

        b_a_red = np.zeros(b_a.shape[:-1])
        b_a_blue = np.zeros(b_a.shape[:-1])

        reddish = b_a[:, :, 0] > 50
        blueish = b_a[:, :, 2] > 10
        b_a_red[reddish] = 1
        b_a_blue[blueish] = 1
        b_a_blue[reddish] = 0

        b.append(np.asarray([b_a_red, b_a_blue]))
      yield torch.FloatTensor(a).to(device), torch.FloatTensor(b).to(device)

  def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
      outputs.append(f"{k}: {metrics[k] / epoch_samples:4f}")

    print(f"{phase}: {', '.join(outputs)}")

  def reverse_transform(inp):
    inp = inp.transpose((1,2,0))
    inp = inp * 255.0
    inp = np.clip(inp,0,255).astype(np.uint8)
    return inp

  def transform(inp):
    inp = inp / 255.0
    inp = np.clip(inp,0,1)
    inp = inp.transpose((2,0,1))
    return inp

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10

  d_l = iter(data_loader(device))

  since = time.time()

  for epoch in range(num_updates):
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        #for param_group in optimizer.param_groups:
        #  print("LR", param_group['lr'])

        model.train()  # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode

      metrics = defaultdict(float)
      epoch_samples = 0

      inputs, labels = next(d_l)
      optimizer.zero_grad()

      with torch.set_grad_enabled(phase == 'train'):
        outputs = model(inputs)
        loss = calc_loss(outputs, labels, metrics)

        if phase == 'train':  # backward + optimize only if in training phase
          loss.backward()
          optimizer.step()

      # statistics
      epoch_samples += inputs.size(0)

      print_metrics(metrics, epoch_samples, phase)
      epoch_loss = metrics['loss'] / epoch_samples

      # deep copy the model
      if phase == 'val' and epoch_loss < best_loss:
        print("saving best model")
        best_loss = epoch_loss
        best_model_wts = copy.deepcopy(model.state_dict())

    if epoch_loss< 0.97:
      break

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:4f}')

  model.load_state_dict(best_model_wts)  # load best model weights

  model.eval()  # Set model to evaluate mode

  inputs, labels = next(d_l)

  pred = model(inputs).data.cpu().numpy()

  input_images_rgb = [reverse_transform(x) for x in inputs.cpu().numpy()]

  l = labels.cpu().numpy()
  target_masks_rgb = [helper.masks_to_color_img(x) for x in l]  # Map each channel (i.e. class) to each color
  pred_rgb = [helper.masks_to_color_img(x) for x in pred]

  helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])
  plt.show()

  return model


if __name__ == '__main__':

  torch.manual_seed(2)

  env = ObservationWrapper(environment_name="",
                           connect_to_running=True)
  env.seed(3)

  # _agent = C.AGENT_TYPE(C)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = UNet(2, depth=4, merge_mode='concat')
  model = model.to(device)

  optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

  try:
    train_model(model, optimizer_ft, exp_lr_scheduler, device=device)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()

  env.close()
