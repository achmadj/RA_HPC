import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from icecream import ic
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
from convNd import convNd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from conv6d_h2 import NN


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Data preparation
with h5py.File('/clusterfs/students/achmadjae/RA/03_deep6D/data_helium/h2.h5', 'r') as f:
    data = f['potentials'][:]
    label = f['labels'][:]

ic(data.shape, label.shape)

L = data.shape[1]
N = data.shape[0]
batch_size = 25
x = torch.tensor(data.reshape(N, 1, L, L, L, L, L, L), dtype=torch.float32)
y = torch.tensor(label.reshape(N, 1), dtype=torch.float32)

dataset = torch.utils.data.TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=23)
val_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=23)

model = NN().to(device)
# model.save('conv6d_h2.pt')
checkpoint = torch.load('/clusterfs/students/achmadjae/RA/03_deep6D/convNd/lightning_logs/version_0/checkpoints/epoch=9-step=180.ckpt')
weight = checkpoint['state_dict']
model.load_state_dict(weight)


# # predict with dataloader
model.eval()
X_PRED = []

for batch in dataloader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_hat = model(x)
    loss = model.criterion(y_hat, y)
    ic(loss.item())
    break
x, y = next(iter(dataloader))
X_ = x.to(device)
y_ = model(X_).cpu()
targets = y

# def plot(x, y):
#     X = x.detach().numpy()
#     Y = y.cpu().numpy()
#     c = np.abs(X - Y)
#     plt.scatter(X, Y, s=5, c=c, cmap='inferno')
#     maks = np.max([np.max(X), np.max(Y)])
#     plt.xlim(np.min([np.min(X), np.min(Y)]), np.max([np.max(X), np.max(Y)]))
#     plt.ylim(np.min([np.min(X), np.min(Y)]), np.max([np.max(X), np.max(Y)]))
#     plt.xlabel('predicted')
#     plt.ylabel('actual')
#     plt.savefig('pred_h2.png')
#     plt.show()

# plot(y_, targets)