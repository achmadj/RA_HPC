import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import numpy as np 
# import scipy.sparse as sp
# import scipy.sparse.linalg as linalg
import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as linalg
import cupy as cp
import h5py
import os
from icecream import ic
import sys


class Solver:
  def __init__(self, L, limit, N, mode='sho'):
    self.L = L
    self.limit = limit
    self.N = N
    self.mode = mode

  def getMesh(self):
    """
    return:
      mesh: meshgrid of x, y, z
      h: step size
    """
    x = np.linspace(-self.limit, self.limit, self.L)
    h = x[1]-x[0]
    mesh = np.meshgrid(x, x, x)
    return mesh, h
  
  def getRandom(self):
    """
    return:
      param: random parameters
    """
    if self.mode == 'sho':
      kx = np.random.rand(self.N) * 0.36
      ky = np.random.rand(self.N) * 0.36
      kz = np.random.rand(self.N) * 0.36
      cx = -8 + (np.random.rand(self.N)) * 8
      cy = -8 + (np.random.rand(self.N)) * 8
      cz = -8 + (np.random.rand(self.N)) * 8

      return kx, ky, kz, cx, cy, cz

    elif self.mode == 'dwig':
      A1 = 2 + np.random.random(self.N)*2
      A2 = 2 + np.random.random(self.N)*2
      A3 = 2 + np.random.random(self.N)*2
      cx1 = -8 + np.random.random(self.N)*16
      cx2 = -8 + np.random.random(self.N)*16
      cx3 = -8 + np.random.random(self.N)*16
      cy1 = -8 + np.random.random(self.N)*16
      cy2 = -8 + np.random.random(self.N)*16
      cy3 = -8 + np.random.random(self.N)*16
      cz1 = -8 + np.random.random(self.N)*16
      cz2 = -8 + np.random.random(self.N)*16
      cz3 = -8 + np.random.random(self.N)*16
      kx1 = 1.6 + np.random.random(self.N)*6.4
      kx2 = 1.6 + np.random.random(self.N)*6.4
      kx3 = 1.6 + np.random.random(self.N)*6.4
      ky1 = 1.6 + np.random.random(self.N)*6.4
      ky2 = 1.6 + np.random.random(self.N)*6.4
      ky3 = 1.6 + np.random.random(self.N)*6.4
      kz1 = 1.6 + np.random.random(self.N)*6.4
      kz2 = 1.6 + np.random.random(self.N)*6.4
      kz3 = 1.6 + np.random.random(self.N)*6.4

      return A1, A2, A3, cx1, cy1, cz1, cx2, cy2, cz2, cx3, cy3, cz3, kx1, ky1, kz1, kx2, ky2, kz2, kx3, ky3, kz3
    
    elif self.mode == 'iw':
      E = 0.1 + np.random.random(self.N)*0.8
      term1 = 2*E / np.pi**2
      Lxs = []
      Lys = []
      Lzs = []
      cxs = []
      cys = []
      czs = []
      for term in term1:
        cx = -8 + (np.random.rand() * 16)
        cy = -8 + (np.random.rand() * 16)
        cz = -8 + (np.random.rand() * 16)
        Ly = 0.001
        Lz = 0.001
        while (Ly**(-2) + Lz**(-2)) > term:
          Ly, Lz = self.RandomL()
        Lx = 1/np.sqrt(term - Ly**(-2) - Lz**(-2))
        Lxs.append(Lx)
        Lys.append(Ly)
        Lzs.append(Lz)
        cxs.append(cx)
        cys.append(cy)
        czs.append(cz)

      return E, Lxs, Lys, Lzs, cxs, cys, czs
    
  def T(self, h):
    diag = np.ones([self.L])
    diags = np.array([diag, -2*diag, diag])
    D = sp.spdiags(diags, np.array([-1, 0, 1]), self.L, self.L)/h**2
    D1 = sp.kronsum(D,D)
    D1 = sp.kronsum(D1,D)
    return -0.5 * D1
      
  def V_sho(self, mesh, param):
    kx, ky, kz, cx, cy, cz = param
    (x,y,z) = mesh
    potential = 0.5 * (kx*(x-cx)**2 + ky*(y-cy)**2 + kz*(z-cz)**2)
    V = sp.diags(potential.reshape(self.L**3),(0), )
    return V, potential

  def V_dwig(self, mesh, param):
    A1, A2, A3, cx1, cy1, cz1, cx2, cy2, cz2, cx3, cy3, cz3, kx1, ky1, kz1, kx2, ky2, kz2, kx3, ky3, kz3 = param
    (x,y,z) = mesh
    potential = \
      - A1 * np.exp(-((x - cx1) / kx1)**2 - ((y - cy1) / ky1)**2 - ((z - cz1) / kz1)**2) \
      - A2 * np.exp(-((x - cx2) / kx2)**2 - ((y - cy2) / ky2)**2 - ((z - cz2) / kz2)**2) \
      - A3 * np.exp(-((x - cx3) / kx3)**2 - ((y - cy3) / ky3)**2 - ((z - cz3) / kz3)**2)
    V = sp.diags(potential.reshape(self.L**3),(0), )
    return V, potential
  
  def RandomL(self):
    Ly = 1 + np.random.rand() * 14
    Lz = 1 + np.random.rand() * 14
    return Ly, Lz

  def V_iw(self, mesh, param):
    (x,y,z) = mesh
    E, Lx, Ly, Lz, cx, cy, cz = param
    upperX = + Lx
    lowerX = - Lx
    upperY = + Ly
    lowerY = - Ly
    upperZ = + Lz
    lowerZ = - Lz

    potential = np.zeros((self.L, self.L, self.L))
    for i in range(self.L):
      for j in range(self.L):
        for k in range(self.L):
          if (lowerX < x[i,j,k] <= upperX) and (lowerY < y[i,j,k] <= upperY) and (lowerZ < z[i,j,k] <= upperZ):
            potential[i,j,k] = 0
          else:
            potential[i,j,k] = 20
    
    V = sp.diags(potential.reshape(self.L**3),(0), )
    
    return V, potential, E

  def solve(self):
    """
    return:
      energies: energies of the images
      imgs: images
    """
    mesh, h = self.getMesh()
    T = self.T(h)
    (param) = self.getRandom()
    param = np.array(param).T
    energies = np.zeros((self.N, 1))
    imgs = np.zeros((self.N, self.L, self.L, self.L))
      
    for idx in range(self.N):
      if self.mode=='iw':
        # print(idx, end='\r', flush=True)
        V, img, E = self.V_iw(mesh, param[idx])
        energies[idx] = E
        imgs[idx] = img
        
      else:
        V, img = self.V_sho(mesh, param[idx]) if self.mode == 'sho' else self.V_dwig(mesh, param[idx])
        H = T + V
        E = linalg.eigsh(H, k=1, which='SA', return_eigenvectors=False)
        E0 = E[0]
        energies[idx, 0] = np.real(E0.get())
        imgs[idx] = img
        del H, E, E0
    # remove unused variable
    del mesh, h, T, param, V, img,
    return energies, imgs
  
  def create_train_data(self, filename, num_augmentations=0.2):
    """
    param:
      filename: name of the file to save the data 
      num_augmentations: number of augmentations per image. Default is range between 0 and 1. If int is given, it is the number of augmentations per image.
    return:
      energies: energies of the images after augmentation (e0, e1, kinetic)
      imgs: images
    """
    energies, imgs = self.solve()
    # generate random indeks sebanyak num_augmentations
    if num_augmentations < 1:
      num_augmentations = int(self.N*num_augmentations)
    
    if self.mode != 'iw':
      idx = np.random.randint(0, self.N, num_augmentations)
      addition_data = np.zeros((num_augmentations, self.L, self.L, self.L))
      rotations = [90, 180, 270]
      idx_ax = [0, 1, 2]
      axes = [(0,1), (1,2), (0,2)]

      for index, i in enumerate(idx):
        # rotate
        rot = np.random.choice(rotations)
        idx_axis = np.random.choice(idx_ax)
        addition_data[index] = rotate(imgs[i], angle=rot, axes=axes[idx_axis], reshape=False)

      energies = np.concatenate((energies, energies[idx]))
      imgs = np.concatenate((imgs, addition_data))

      # del unused variable
      del addition_data, rotations, axes, idx

    with h5py.File(filename, 'w') as hf:
      hf.create_dataset('feature', data=imgs)
      hf.create_dataset('target', data=energies)
    return energies, imgs
  
  def create_test_data(self, filename, num_data):
    """
    param:
      filename: name of the file to save the data 
      num_data: number of data to be generated
    return:
      energies: energies of the images
      imgs: images
    """
    if num_data<1:
      self.N = int(self.N*num_data)
    else:
      self.N = num_data

    energies, imgs = self.solve()
    with h5py.File(filename, 'w') as hf:
      hf.create_dataset('feature', data=imgs)
      hf.create_dataset('target', data=energies)
    return energies, imgs
  
def random_split(X, y, val_split=0.2):
  N = X.shape[0]
  # idx = np.random.permutation(N)
  # train_idx = idx[:int(train_split*N)]
  # val_idx = idx[int(train_split*N):int((train_split+val_split)*N)]
  idx = np.random.randint(0, N, int(val_split*N))
  val_idx = idx
  train_idx = np.delete(np.arange(N), idx)
  return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


if __name__ == "__main__":
  L = 40
  limit = 20
  N_train = 20000
  N_test = 5000
  mode = sys.argv[1]
  ic(mode)
  train_filename = f'{mode}_train_{L}_15000.h5'
  test_filename = f'{mode}_test_{L}_5000.h5'
  S = Solver(L, limit, N_train, mode)
  S.create_train_data(f'/clusterfs/students/achmadjae/RA/02_deep3D/data/{train_filename}', 5000)
  S.create_test_data(f'/clusterfs/students/achmadjae/RA/02_deep3D/data/{test_filename}', N_test)
  ic('done')
