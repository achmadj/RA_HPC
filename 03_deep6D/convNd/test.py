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
from icecream import ic
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Data preparation
with h5py.File('/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/data_helium/hehe.h5', 'r') as f:
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

# Create Model with pytorch lightning module
class NN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv6d1 = convNd( in_channels=1, out_channels=128, num_dims=6, kernel_size=3, stride=(1, 1, 1, 1, 1, 1), padding=0)
        self.conv6d2 = convNd( in_channels=128, out_channels=256, num_dims=6, kernel_size=3, stride=(1, 1, 1, 1, 1, 1), padding=0)
        self.conv6d3 = convNd( in_channels=256, out_channels=64, num_dims=6, kernel_size=3, stride=(1, 1, 1, 1, 1, 1), padding=0)
        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4096, 1)
        self.criterion = nn.L1Loss()

    def forward(self, x):
        x = self.relu(self.conv6d1(x))
        x = self.relu(self.conv6d2(x))
        x = self.relu(self.conv6d3(x))
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def on_validation_epoch_end(self):
        print('val_loss: ', self.trainer.callback_metrics['val_loss'], flush=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, patience=5, verbose=True, min_lr=1e-6, factor=0.1)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'train_loss'}


# Train model
model = NN().to(device)
# ckpt_path = '/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd/lightning_logs/version_0/checkpoints/epoch=15-step=288.ckpt'
ckpt_path = '/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/convNd/lightning_logs/version_2/checkpoints/epoch=15-step=288.ckpt'
checkpoint = torch.load(ckpt_path)
weight = checkpoint['state_dict']
model.load_state_dict(weight)

y_true = []
y_pred = []
model.eval()
for features, targets in tqdm(val_dataloader):
    features = features.to(device)
    targets = targets.to(device)
    y_true.append(targets.cpu().detach().numpy())
    y_pred.append(model(features).cpu().detach().numpy())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

plt.plot(y_true, '.', label='True', color='blue')
plt.plot(y_pred, '.', label='Predicted', color='red')
plt.legend()
plt.savefig('true_vs_pred_hehe.png')
plt.show()