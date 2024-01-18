import torch
from torch import nn
from torch.utils.data import DataLoader
# import Functional
from torch.nn import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from icecream import ic

from convNd import convNd

import pytorch_lightning as pl

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

# Data preparation
with h5py.File('/clusterfs/students/achmadjae/RA/data_helium/helium.h5', 'r') as f:
    data = f['potentials'][:]
    label = f['labels'][:]

L = data.shape[1]
N = data.shape[0]
batch_size = 10

x = torch.tensor(data.reshape(N, 1, L, L, L, L, L, L), dtype=torch.float32).cuda()
y = torch.tensor(label.reshape(N, 1), dtype=torch.float32).cuda()

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define median absolute error
class MedianAbsoluteError(nn.Module):
    def __init__(self):
        super(MedianAbsoluteError, self).__init__()
    def forward(self, y_hat, y):
        return torch.median(torch.abs(y_hat - y))

# Create model
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.conv6d1 = convNd(
            in_channels=1,
            out_channels=64,
            num_dims=6,
            kernel_size=3,
            stride=(1, 1, 1, 1, 1, 1),
            padding=0,
        )

        self.conv6d2 = convNd(
            in_channels=64,
            out_channels=128,
            num_dims=6,
            kernel_size=3,
            stride=(1, 1, 1, 1, 1, 1),
            padding=0,
        )

        self.conv6d3 = convNd(
            in_channels=128,
            out_channels=64,
            num_dims=6,
            kernel_size=3,
            stride=(1, 1, 1, 1, 1, 1),
            padding=0,
        )

        self.conv6d4 = convNd(
            in_channels=64,
            out_channels=32,
            num_dims=6,
            kernel_size=3,
            stride=(1, 1, 1, 1, 1, 1),
            padding=0,
        )

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.relu(self.conv6d1(x))
        x = self.relu(self.conv6d2(x))
        x = self.relu(self.conv6d3(x))
        x = self.relu(self.conv6d4(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

# MCOC
model = NN().cuda()
# Load model
# model.load_state_dict(torch.load('/clusterfs/students/achmadjae/RA/convNd/MODEL.pt'))
criterion = MedianAbsoluteError()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True, min_lr=1e-6, factor=0.1)
# callback = lambda x: print(f'Loss: {x}')

# Training
model.train()
epoch = 300
for ep in range(epoch):
    cost = 0
    for batch in dataloader: #tqdm(dataloader, leave=False, desc=f'Epoch {ep}/{epoch}'):
        feature, target = batch
        y_hat = model(feature)
        loss = criterion(y_hat, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss.item() * feature.shape[0]
    cost /= len(dataset)
    scheduler.step(cost)
    
    actual = round(target[np.int_(np.random.rand()*batch_size)].item(), 6)
    predicted = round(y_hat[np.int_(np.random.rand()*batch_size)].item(), 6)
    print(f'Loss: {round(cost, 6)} with random value of target: {actual} and predicted value: {predicted}', flush=True)

# Evaluation
model.eval()
with torch.no_grad():
    y_ = []
    targets = []
    for batch in tqdm(dataloader):
        feature, target = batch
        pred = model(feature)
        y_.append(pred)
        targets.append(target)

# Save model
torch.save(model.state_dict(), 'model.pt')

# Plot
y_ = torch.cat(y_).flatten()
targets = torch.cat(targets).flatten()

def plot(x, y):
    X = x.cpu().numpy()
    Y = y.cpu().numpy()
    c = np.abs(X - Y)
    plt.scatter(X, Y, s=5, c=c, cmap='inferno')
    maks = np.max([np.max(X), np.max(Y)])
    plt.xlim(-0.5, maks)
    plt.ylim(-0.5, maks)
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.savefig('pred.png')
    plt.show()

plot(y_, targets)