import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from icecream import ic
import pytorch_lightning as pl
from pytorch_lightning.utilities.model_summary import ModelSummary
# from convNd import convNd
import h5py
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Data preparation
with h5py.File('/mgpfs/home/ajaelani/_scratch/conv_6d/RA_HPC/03_deep6D/data_helium/hehe.h5', 'r') as f:
    # data = f['potentials'][:]
    label = f['labels'][:]

print(label)