#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import copy
import time

import warg
from PIL import Image
from experiments.segmentation import helper
from experiments.segmentation.losses.accum import calculate_accum_loss
from experiments.segmentation.models.segnet_arch import SegNetArch
from experiments.segmentation.unet.aeunet import AEUNet
from experiments.segmentation.unet.unet_small import UNetSmall
from neodroid.wrappers.observation_wrapper.observation_wrapper import (CameraObservationWrapper, 
  neodroid_batch_data_iterator)

from utilities import reverse_channel_transform

__author__ = 'cnheider'

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

tqdm.monitor_interval = 0
interrupted_path= 'INTERRUPTED_BEST.pth'

def get_str(metrics, phase):
  outputs = []
  for k in metrics.keys():
    outputs.append(f"{k}: {metrics[k]:4f}")

  return f"{phase}: {', '.join(outputs)}"


def train_model(model, data_iterator, optimizer, scheduler, num_updates=25000):
  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10
  since = time.time()

  try:
    sess = tqdm(range(num_updates),leave=False)
    for epoch in sess:
      # Each epoch has a training and validation phase
      for phase in ['train', 'val']:
        if phase == 'train':
          scheduler.step()
          model.train()  # Set model to training mode
        else:
          model.eval()  # Set model to evaluate mode

        inputs, labels = next(data_iterator)
        optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
          outputs, reconstruction = model(inputs)
          ret = calculate_accum_loss(outputs, labels, reconstruction, inputs)

          if phase == 'train':  # backward + optimize only if in training phase
            ret.loss.backward()
            optimizer.step()

        epoch_loss = ret.metrics['total_l']

        if phase == 'val' and epoch_loss < best_loss:
          best_loss = epoch_loss
          best_model_wts = copy.deepcopy(model.state_dict())

      sess.set_description_str(f'{epoch} - {get_str(ret.metrics, phase)}')

      if epoch_loss < 0.1:
        break

  except KeyboardInterrupt:
    print('Interrupt')
  finally:
    model.load_state_dict(best_model_wts)  # load best model weights
    torch.save(model.state_dict(), interrupted_path)

  time_elapsed = time.time() - since
  print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
  print(f'Best val loss: {best_loss:4f}')

  return model


def test_model(model, data_iterator):
  model.load_state_dict(torch.load(interrupted_path))

  model.eval()  # Set model to evaluate mode

  inputs, labels = next(data_iterator)[:3]

  pred, recon = model(inputs)
  pred = pred.data.cpu().numpy()
  recon = recon.data.cpu().numpy()
  l = labels.cpu().numpy()
  inputs = inputs.cpu().numpy()

  input_images_rgb = [reverse_channel_transform(x) for x in inputs]
  target_masks_rgb = [helper.masks_to_color_img(reverse_channel_transform(x)) for x in l]
  pred_rgb = [helper.masks_to_color_img(reverse_channel_transform(x)) for x in pred]
  pred_recon = [reverse_channel_transform(x) for x in recon]

  helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb, pred_recon])
  plt.show()



def main():
  env = CameraObservationWrapper()
  torch.manual_seed(2)
  env.seed(3)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  aeu_model = AEUNet(3, start_filters=16, depth=5)
  #aeu_model = UNetSmall(3)
  #aeu_model = SegNetArch(3)
  aeu_model = aeu_model.to(device)

  batch_size = 12

  optimizer_ft = optim.Adam(aeu_model.parameters(), lr=1e-4)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=int(400 // batch_size) + 4, gamma=0.1)

  d_l = iter(neodroid_batch_data_iterator(env,device,batch_size))

  args = argparse.ArgumentParser()
  args.add_argument('-i', action='store_false')
  options = args.parse_args()

  try:
    if options.i:
      aeu_model = train_model(aeu_model, d_l, optimizer_ft, exp_lr_scheduler)
    test_model(aeu_model, d_l)
  except KeyboardInterrupt:
    print('Stopping')
  finally:
    torch.cuda.empty_cache()
    env.close()

    '''
    try:
      sys.exit(0)
    except SystemExit:
      os._exit(0)
    '''


if __name__ == '__main__':
  main()
