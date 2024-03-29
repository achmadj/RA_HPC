import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import numpy as np 
import scipy.sparse as sp
import scipy.sparse.linalg as linalg
# import cupyx.scipy.sparse as sp
# import cupyx.scipy.sparse.linalg as linalg
# import cupy as cp
import h5py
import os
from icecream import ic
import sys
import time

class Solver:
  def __init__(self, L, limit, N, mode='sho', **kwargs):
    for key, value in kwargs.items():
      setattr(self, key, value)
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
      A1 = self.leftA + np.random.random(self.N) * self.rightA
      A2 = self.leftA + np.random.random(self.N) * self.rightA
      A3 = self.leftA + np.random.random(self.N) * self.rightA
      cx1 = self.leftc + np.random.random(self.N)*self.rightc
      cx2 = self.leftc + np.random.random(self.N)*self.rightc
      cx3 = self.leftc + np.random.random(self.N)*self.rightc
      cy1 = self.leftc + np.random.random(self.N)*self.rightc
      cy2 = self.leftc + np.random.random(self.N)*self.rightc
      cy3 = self.leftc + np.random.random(self.N)*self.rightc
      cz1 = self.leftc + np.random.random(self.N)*self.rightc
      cz2 = self.leftc + np.random.random(self.N)*self.rightc
      cz3 = self.leftc + np.random.random(self.N)*self.rightc
      kx1 = self.leftk + np.random.random(self.N)*self.rightk
      kx2 = self.leftk + np.random.random(self.N)*self.rightk
      kx3 = self.leftk + np.random.random(self.N)*self.rightk
      ky1 = self.leftk + np.random.random(self.N)*self.rightk
      ky2 = self.leftk + np.random.random(self.N)*self.rightk
      ky3 = self.leftk + np.random.random(self.N)*self.rightk
      kz1 = self.leftk + np.random.random(self.N)*self.rightk
      kz2 = self.leftk + np.random.random(self.N)*self.rightk
      kz3 = self.leftk + np.random.random(self.N)*self.rightk

      return A1, A2, A3, cx1, cy1, cz1, cx2, cy2, cz2, cx3, cy3, cz3, kx1, ky1, kz1, kx2, ky2, kz2, kx3, ky3, kz3
    
    elif self.mode == 'iw':
      # ic("here")
      E = 0.1 + np.random.random(int(self.N/3))*0.8
      term1 = np.pi**2 / (2 * E)
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
        Ly = Lz = np.inf
        while (Ly**2 + Lz**2) > term:
          Ly, Lz = self.RandomL()
        Lx = np.sqrt(term - Ly**2 - Lz**2)
        Lxs.append(Lx)
        Lys.append(Ly)
        Lzs.append(Lz)
        cxs.append(cx)
        cys.append(cy)
        czs.append(cz)

      E = np.repeat(E, 3)
      Lxs_ = np.array(Lxs + Lys + Lzs)
      Lys_ = np.array(Lys + Lzs + Lxs)
      Lzs_ = np.array(Lzs + Lxs + Lys)
      cx_ = np.array(cxs + cys + czs)
      cy_ = np.array(cys + czs + cxs)
      cz_ = np.array(czs + cxs + cys)

      if self.N%3!=0:
        iw_len = int(self.N/3)*3
        range_ = self.N - iw_len
        idx = np.random.randint(0, iw_len, range_)
        E = np.concatenate((E, E[idx]))
        Lxs_ = np.concatenate((Lxs_, Lxs_[idx]))
        Lys_ = np.concatenate((Lys_, Lys_[idx]))
        Lzs_ = np.concatenate((Lzs_, Lzs_[idx]))
        cx_ = np.concatenate((cx_, cx_[idx]))
        cy_ = np.concatenate((cy_, cy_[idx]))
        cz_ = np.concatenate((cz_, cz_[idx]))

      # delete unreturned variable
      del Lxs, Lys, Lzs
      del Ly, Lz, Lx, term1
      del cxs, cys, czs
      # ic(len(E), len(Lxs_), len(Lys_), len(Lzs_), len(cx_), len(cy_), len(cz_))
      return E, Lxs_, Lys_, Lzs_, cx_, cy_, cz_
    
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
    upperX = Lx
    lowerX = Lx
    upperY = Ly
    lowerY = Ly
    upperZ = Lz
    lowerZ = Lz

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
    if self.mode=='iw':
      energies = np.zeros((self.N, 1))
      
    for idx in range(self.N):
      if self.mode=='iw':
        # print(idx, end='\r', flush=True)
        V, img, E = self.V_iw(mesh, param[idx])
        energies[idx] = E
        imgs[idx] = img
        
      else:
        V, img = self.V_sho(mesh, param[idx]) if self.mode == 'sho' else self.V_dwig(mesh, param[idx])
        H = T + V
        # E, psi = linalg.eigsh(H, k=1, which='SA', return_eigenvectors=False)
        E = linalg.eigsh(H, k=1, which='SA', return_eigenvectors=False)
        E0 = E[0]
        # E1 = E[1]
        # wf = psi[:, 0]
        # kinetic = wf.dot(T.dot(wf.T))
        energies[idx, 0] = np.real(E0)
        # energies[idx, 1] = np.real(E1)
        # energies[idx, 2] = np.real(kinetic)

        imgs[idx] = img
        # del H, E, psi, E0, E1, wf, kinetic
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
    idx = np.random.randint(0, self.N, int(num_augmentations))
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
  N_train = 1000
  N_test = 5000
  mode = 'dwig'
#  ic(mode)
  train_filename = f'{mode}_train_{L}_{N_train}.h5'
  test_filename = f'{mode}_test_{L}_{N_test}.h5'

  LEFTA = np.arange(1, 10, 1)
  RIGHTA = np.arange(1, 10, 1)
  LEFTK = np.arange(-10, -1, 1)
  RIGHTK = np.arange(2, 20, 2)
  LEFTC = np.arange(0.1, 4, 0.1)
  RIGHTC = np.arange(1, 4, 0.1)

  def solve(params):
    leftA, rightA, leftk, rightk, leftc, rightc = params
    now = time.time()
    S = Solver(L, limit, N_train, mode, leftA=leftA, rightA=rightA, leftk=leftk, rightk=rightk, leftc=leftc, rightc=rightc)
    energies, imgs = S.solve()
    del imgs
    print(f'leftA: {leftA} --- rightA: {rightA} --- leftk: {leftk} --- rightk: {rightk} --- leftc: {leftc} --- rightc: {rightc}', end=' ', flush=True)
    print(f'MAX: {np.max(energies)} --- MIN: {np.min(energies)}', flush=True)
    then = time.time()
    print(f'{(then-now):.2f} s', flush=True)

  from multiprocessing import Pool
  from itertools import product

  parameters = product(LEFTA, RIGHTA, LEFTK, RIGHTK, LEFTC, RIGHTC)
  # print(parameters)
  with Pool(32) as pool:
    print(pool.map(solve, parameters), flush=True)

  # for leftA in LEFTA:
  #   for rightA in RIGHTA:
  #     for leftk in LEFTK:
  #       for rightk in RIGHTK:
  #         for leftc in LEFTC:
  #           for rightc in RIGHTC:
  #             now = time.time()
  #             S = Solver(L, limit, N_train, mode, leftA=leftA, rightA=rightA, leftk=leftk, rightk=rightk, leftc=leftc, rightc=rightc)
  #             energies, imgs = S.solve()
  #             del imgs
  #             print(f'leftA: {leftA} --- rightA: {rightA} --- leftk: {leftk} --- rightk: {rightk} --- leftc: {leftc} --- rightc: {rightc}', end=' ', flush=True)
  #             print(f'MAX: {np.max(energies)} --- MIN: {np.min(energies)}', flush=True)
  #             then = time.time()
  #             print(f'{(then-now):.2f} s', flush=True)

