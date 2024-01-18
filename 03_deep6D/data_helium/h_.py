import cupyx.scipy.sparse as sp
import cupyx.scipy.sparse.linalg as linalg
import cupy as np
import numpy
from tqdm.auto import tqdm

L = 10
h = 0.4
limit = (L-1)*h/2
charge = 2
filename = "helium.h5"
N = 1000
potentials = np.zeros([N, L, L, L, L, L, L, 1])
labels = np.zeros([N, 1])
kinetics = np.zeros([N, 1])

for i in tqdm(range(N)):
  # mesh
  x = np.linspace(-limit, limit, L)
  dx = x[1]-x[0] #grid spacing
  (X1, X2, Y1, Y2, Z1, Z2) = np.meshgrid(x, x, x, x, x, x, indexing="ij")
  energy = -100

  # coordinate
  while energy < -2.90372 or energy > -2.84000:
    cx, cy, cz = np.random.uniform(-h/2, h/2, 3)
    while np.sqrt(cx**2 + cy**2 + cz**2) > h/2:
      cx, cy, cz = np.random.uniform(-h/2, h/2, 3)

    R1 = np.sqrt((X1-cx)**2 + (Y1-cy)**2 + (Z1-cz)**2)
    R2 = np.sqrt((X2-cx)**2 + (Y2-cy)**2 + (Z2-cz)**2)
    r12 = np.sqrt((X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2)

    # kinetic
    diag = np.ones([L])
    diags = np.array([diag, -2*diag, diag])
    D = sp.spdiags(diags, np.array([-1, 0, 1]), L, L)/dx/dx
    D1 = sp.kronsum(sp.kronsum(sp.kronsum(sp.kronsum(sp.kronsum(D,D), D), D), D), D)
    T = -0.5 * D1

    with numpy.errstate(divide='ignore'):
      # attraction
      attraction_1 = np.where(R1==0, 0, -charge/R1)    #-2/np.sqrt((X1)**2 + (Y1)**2 + (Z1)**2)
      attraction_2 = np.where(R2==0, 0, -charge/R2)    #-2/np.sqrt((X2)**2 + (Y2)**2 + (Z2)**2)

      # repulsion
      repulsion = np.where(r12==0, 0, 1/r12)    #1/np.sqrt((X1-X2)**2 + (Y1-Y2)**2 + (Z1-Z2)**2)

    # potential
    V = sp.diags((attraction_1 + attraction_2 + repulsion).reshape(L**6),(0), )
    # solve
    H = T + V
    energy, psi = linalg.eigsh(H, k=1, which='SA', return_eigenvectors=True)
  potential = (attraction_1 + attraction_2 + repulsion)
  potentials[i] = potential.reshape([L, L, L, L, L, L, 1])
  labels[i] = energy
  kinetics[i] = psi.T @ T @ psi

import h5py

with h5py.File(filename, 'w') as f:
  f.create_dataset('potentials', data=potentials.get())
  f.create_dataset('labels', data=labels.get())
  f.create_dataset('kinetics', data=kinetics.get())
