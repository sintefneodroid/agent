#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import copy
import time
from collections import defaultdict

import torchvision.transforms as T
from PIL import Image

from experiments.segmentation import helper
from experiments.segmentation.loss import calc_loss
from experiments.segmentation.unet import UNet
from neodroid.wrappers.observation_wrapper.observation_wrapper import ObservationWrapper

__author__ = 'cnheider'

import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

tqdm.monitor_interval = 0

resize = T.Compose([
  #T.ToPILImage(),
                   # T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def train_model(model, optimizer, scheduler, batch_size=8, num_epochs=25, device='cpu'):
  def data_loader(device):
    while True:
      a = []
      b = []
      while len(a) < batch_size:
        a.append(transform(np.array(Image.open(env.observer('RGBCameraObserver').observation_value))))
        b.append(transform(np.array(Image.open(env.observer('MatId').observation_value))))
      yield torch.Tensor(a).to(device), torch.Tensor(b).to(device)

  def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
      outputs.append(f"{k}: {metrics[k] / epoch_samples:4f}")

    print(f"{phase}: {', '.join(outputs)}")

  def reverse_transform(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp

  def transform(inp):
    inp = inp.transpose()
    inp = np.clip(inp, 0, 1)
    inp = (inp /255).astype(np.uint8)

    return inp

  best_model_wts = copy.deepcopy(model.state_dict())
  best_loss = 1e10

  for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    since = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
      if phase == 'train':
        scheduler.step()
        for param_group in optimizer.param_groups:
          print("LR", param_group['lr'])

        model.train()  # Set model to training mode
      else:
        model.eval()  # Set model to evaluate mode

      metrics = defaultdict(float)
      epoch_samples = 0

      for inputs, labels in data_loader(device):
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

    time_elapsed = time.time() - since
    print(f'{time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val loss: {best_loss:4f}')

    model.load_state_dict(best_model_wts)  # load best model weights

    model.eval()  # Set model to evaluate mode

    inputs, labels = next(iter(data_loader(device)))

    pred = model(inputs).data.cpu().numpy()

    input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    target_masks_rgb = [helper.masks_to_color_img(x) for x in labels.cpu().numpy()]
    pred_rgb = [helper.masks_to_color_img(x) for x in pred]

    helper.plot_side_by_side([input_images_rgb, target_masks_rgb, pred_rgb])

    return model


if __name__ == '__main__':

  torch.manual_seed(2)

  env = ObservationWrapper(environment_name="",
                           connect_to_running=True)
  env.seed(3)

  # _agent = C.AGENT_TYPE(C)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  model = UNet(3, depth=2, merge_mode='concat')
  model = model.to(device)

  optimizer_ft = optim.Adam(model.parameters(), lr=1e-4)
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=25, gamma=0.1)

  try:
    train_model(model, optimizer_ft, exp_lr_scheduler,device=device)
  except KeyboardInterrupt:
    print('Stopping')

  torch.cuda.empty_cache()

  env.close()
