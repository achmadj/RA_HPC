import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import datetime
import time
from icecream import ic
import sys

plt.rcParams['text.usetex'] = True

def random_split(X, y, train_split):
  N = X.shape[0]
  idx = np.random.permutation(N)
  train_idx = idx[:int(train_split*N)]
  val_idx = idx[int(train_split*N):]
  return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

def data_preparation(mode, filename, bs=128, N=20000, L=40):
    with h5py.File(filename, 'r') as hf:
      feature = hf['feature'][:]
      target = hf['target'][:, 0]
    ic(feature.shape, target.shape)
    X = torch.tensor(feature.reshape(N, 1, L, L, L), dtype=torch.float32)
    y = torch.tensor(target.reshape(N, 1), dtype=torch.float32)
    del feature, target
    
    if mode == 'train':
      X_train, y_train, X_val, y_val = random_split(X, y, 0.8)
      train_set = TensorDataset(X_train, y_train)
      val_set = TensorDataset(X_val, y_val)

      del X, y
      train_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=16)
      val_loader = DataLoader(val_set, batch_size=bs, shuffle=False, num_workers=16)

      return len(train_set), train_loader, len(val_set), val_loader

    elif mode == 'test':
      test_set = TensorDataset(X, y)
      test_loader = DataLoader(test_set, shuffle=False, num_workers=16)
      return len(test_set), test_loader

# Create Model
class AugNet(nn.Module):
  def __init__(self):
    super().__init__()
    # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    self.reduce1 = nn.Conv3d(1, 64, 4, 2, 1)
    self.nonreduce1_1 = nn.Conv3d(64, 32, 3, 1, 1)
    self.nonreduce1_2 = nn.Conv3d(32, 32, 3, 1, 1)
    self.nonreduce1_3 = nn.Conv3d(32, 32, 3, 1, 1)
    self.reduce2 = nn.Conv3d(32, 16, 4, 2, 1)
    self.nonreduce2_1 = nn.Conv3d(16, 16, 3, 1, 1)
    self.nonreduce2_2 = nn.Conv3d(16, 16, 3, 1, 1)
    self.nonreduce2_3 = nn.Conv3d(16, 16, 3, 1, 1)
    self.reduce3 = nn.Conv3d(16, 8, 4, 2, 1)
    self.nonreduce3_1 = nn.Conv3d(8, 8, 3, 1, 1)
    self.nonreduce3_2 = nn.Conv3d(8, 8, 3, 1, 1)
    self.nonreduce3_3 = nn.Conv3d(8, 8, 3, 1, 1)
    self.flatten = nn.Flatten()
    self.relu = nn.ReLU()
    self.fc = nn.Linear(1000, 1)

  def forward(self, x):
    x = self.relu(self.reduce1(x))
    x = self.relu(self.nonreduce1_1(x))
    x = self.relu(self.nonreduce1_2(x))
    x = self.relu(self.nonreduce1_3(x))
    x = self.relu(self.reduce2(x))
    x = self.relu(self.nonreduce2_1(x))
    x = self.relu(self.nonreduce2_2(x))
    x = self.relu(self.nonreduce2_3(x))
    x = self.relu(self.reduce3(x))
    x = self.relu(self.nonreduce3_1(x))
    x = self.relu(self.nonreduce3_2(x))
    x = self.relu(self.nonreduce3_3(x))
    x = self.flatten(x)
    x = self.fc(x)
    return x

# Training function
def loop_fn(mode, dataset, dataloader, model, criterion, optimizer, scheduler=None):
  if mode == 'train':
    model.train()
  elif mode == 'val':
    model.eval()
  else:
    raise ValueError('mode must be either train or val')

  running_loss = 0.0
  for idx, (feature, target) in enumerate(dataloader):
    feature = feature.to('cuda')
    target = target.to('cuda')
    with torch.set_grad_enabled(mode == 'train'):
      output = model(feature)
      loss = criterion(output, target)
      if mode == 'train':
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    running_loss += loss.item() * feature.size(0)
    del feature, target
    torch.cuda.empty_cache()

  epoch_loss = running_loss / dataset
  scheduler.step(epoch_loss)
  return epoch_loss

def train(model, criterion, optimizer, scheduler, NOW, train_set=None, train_loader=None, val_set=None, val_loader=None, max_epoch=300):
  best_loss = np.inf
  for epoch in range(max_epoch):
    now = time.time()
    train_loss = loop_fn('train', train_set, train_loader, model, criterion, optimizer, scheduler)
    val_loss = loop_fn('val', val_set, val_loader, model, criterion, optimizer, scheduler)
    then = time.time()
    print(f'Epoch: {epoch+1}/{max_epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | {(then-now):.3f} s', flush=True)
    if val_loss < best_loss:
      best_loss = val_loss
      torch.save(model.state_dict(), f'/clusterfs/students/achmadjae/RA/02_deep3D/model/model_{NOW}.pt')

def test(model, NOW, val_loader=None):
  model.load_state_dict(torch.load(f'/clusterfs/students/achmadjae/RA/02_deep3D/model/model_{NOW}.pt'))
  model.eval()
  with torch.no_grad():
    y_true = []
    y_pred = []
    for idx, (feature, target) in enumerate(val_loader):
      feature = feature.to('cuda')
      target = target.to('cuda')
      y_hat = model(feature)
      y_true.append(target.cpu().numpy().flatten())
      y_pred.append(y_hat.cpu().numpy().flatten())

  y_true = np.concatenate(y_true)
  y_pred = np.concatenate(y_pred)

  c = np.abs(y_pred - y_true)
  ic(np.mean(c))
  plt.scatter(y_true, y_pred, s=10, c=c, cmap='viridis_r', rasterized=True)
  plt.colorbar()
  plt.xlabel('True')
  plt.ylabel('Pred')
  MIN = np.min([y_true, y_pred])
  MAX = np.max([y_true, y_pred])
  # plt.xlim(MIN, MAX)
  # plt.ylim(MIN, MAX)
  plt.savefig(f'/clusterfs/students/achmadjae/RA/02_deep3D/fig/plot_{NOW}.pdf', bbox_inches='tight')
  plt.tight_layout()
  plt.show()

if __name__ == '__main__':
  NOW = sys.argv[1]
  mode = sys.argv[2]
  # MCOC
  model = AugNet().to('cuda')
  criterion = nn.L1Loss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
  scheduler = ReduceLROnPlateau(
  optimizer, mode='min', factor=0.1, patience=5, min_lr=1e-6, verbose=True)

  if mode == "train":
    filename = f'/clusterfs/students/achmadjae/RA/02_deep3D/data/{NOW}_train_40_15000.h5'
    train_set, train_loader, val_set, val_loader = data_preparation(mode, filename)
    train(model, criterion, optimizer, scheduler, NOW, train_set, train_loader, val_set, val_loader, max_epoch=1000)
  elif mode=="test":
    filename = f'/clusterfs/students/achmadjae/RA/02_deep3D/data/{NOW}_test_40_5000.h5'
    test_set, test_loader = data_preparation(mode, filename, N=5000)
    test(model, NOW, val_loader=test_loader)
