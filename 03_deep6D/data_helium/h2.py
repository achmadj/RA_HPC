import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as linalg
import cupy as np
# import numpy as np
from tqdm.auto import tqdm
from icecream import ic

# grid
X = np.arange(0.5, 5, 0.01)
g = 8
h1 = 0.9
g3 = g ** 6
r0 = h1 * (g - 1) / 2


filename = "h2.h5"
potentials = np.zeros([len(X), g, g, g, g, g, g, 1])
labels = np.zeros([len(X), 1])
kinetics = np.zeros([len(X), 1])
bond_length = np.zeros([len(X), 1])

for idx, b in tqdm(enumerate(X)):
  # location of nuclei
  xa = np.array(-b / 2)
  ya = np.array(0)
  za = np.array(0)
  xb = np.array(b / 2)
  yb = np.array(0)
  zb = np.array(0)

  # electron coordinates
  p = np.linspace(-r0, r0, g)
  h = p[1] - p[0]
  X1, X2, Y1, Y2, Z1, Z2 = np.meshgrid(p, p, p, p, p, p)
  X1 = X1.flatten()
  Y1 = Y1.flatten()
  Z1 = Z1.flatten()
  X2 = X2.flatten()
  Y2 = Y2.flatten()
  Z2 = Z2.flatten()
  R1 = np.sqrt(X1 ** 2 + Y1 ** 2 + Z1 ** 2)
  R2 = np.sqrt(X2 ** 2 + Y2 ** 2 + Z2 ** 2)
  r12 = np.abs(np.sqrt((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2))
  
  diag = np.ones([g])
  diags = np.array([diag, -2*diag, diag])
  D = sp.spdiags(diags, np.array([-1, 0, 1]), g, g)/h**2
  D1 = sp.kronsum(sp.kronsum(sp.kronsum(sp.kronsum(sp.kronsum(D,D), D), D), D), D)
  T = -0.5 * D1
  # Potential energy
  va1 = -1 / np.sqrt((xa - X1) ** 2 + (ya - Y1) ** 2 + (za - Z1) ** 2)
  va1[np.isinf(va1)] = 0
  va2 = -1 / np.sqrt((xa - X2) ** 2 + (ya - Y2) ** 2 + (za - Z2) ** 2)
  va2[np.isinf(va2)] = 0
  vb1 = -1 / np.sqrt((xb - X1) ** 2 + (yb - Y1) ** 2 + (zb - Z1) ** 2)
  vb1[np.isinf(vb1)] = 0
  vb2 = -1 / np.sqrt((xb - X2) ** 2 + (yb - Y2) ** 2 + (zb - Z2) ** 2)
  vb2[np.isinf(vb2)] = 0
  vx = 1 / r12
  vx[np.isinf(vx)] = 0
  vn = 1 / np.sqrt((xa - xb) ** 2 + (ya - yb) ** 2 + (za - zb) ** 2)
  v = va1 + va2 + vb1 + vb2 + vx + vn
  v[np.isinf(v)] = 0
  U = sp.diags(v, 0, (g3, g3))
  # Hamiltonian and diagonalization
  Hr = T + U
  E, wf = linalg.eigsh(Hr, 1, which='SA')

  potentials[idx] = v.reshape([g, g, g, g, g, g, 1])
  labels[idx] = E[0]
  kinetics[idx] = wf.T @ T @ wf
  bond_length[idx] = b

import h5py

with h5py.File(filename, 'w') as f:
  f.create_dataset('potentials', data=potentials.get())
  f.create_dataset('labels', data=labels.get())
  f.create_dataset('kinetics', data=kinetics.get())
  f.create_dataset('bond_length', data=bond_length.get())