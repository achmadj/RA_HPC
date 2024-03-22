import torch
from torch import nn
from torch.utils.data import DataLoader
from icecream import ic
import h5py
import numpy as np
from conv6d_h2 import NN
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Data preparation
with h5py.File('/clusterfs/students/achmadjae/RA/03_deep6D/data_helium/h2.h5', 'r') as f:
    data = f['potentials'][:]
    label = f['labels'][:]
    bond_length = f['bond_length'][:]

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
checkpoint = torch.load('/clusterfs/students/achmadjae/RA/03_deep6D/convNd/lightning_logs/version_1/checkpoints/epoch=62-step=1134.ckpt')
weight = checkpoint['state_dict']
model.load_state_dict(weight)

# # predict with dataloader
model.eval()
Y_PRED = []
Y_TRUE = []

for batch in val_dataloader:
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    y_hat = model(x)
    Y_PRED.append(y_hat.cpu().detach().numpy())
    Y_TRUE.append(y.cpu().detach().numpy())

Y_PRED = np.concatenate(Y_PRED, axis=0)
Y_TRUE = np.concatenate(Y_TRUE, axis=0)

def plot(Y_PRED, Y_TRUE):
    c = np.abs(Y_PRED - Y_TRUE)
    # plt.scatter(bond_length, Y_TRUE, s=5, color='blue', label='actual')
    plt.scatter(bond_length, Y_PRED, s=5, color='red', label='predicted')
    plt.xlabel('Bond Length (Bohr)')
    plt.ylabel('Energy (Hartree)')
    # plt.legend()
    plt.savefig('/clusterfs/students/achmadjae/RA/03_deep6D/convNd/preds_h2.png')
    plt.show()

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

plot(Y_PRED, Y_TRUE)