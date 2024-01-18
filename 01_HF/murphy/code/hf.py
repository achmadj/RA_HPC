import numpy as np
from scipy import linalg 

scf_max_iter = 20 # maximum number of SCF iterations

print("")

## pass in the number of electrons N, the nuclear repulsion energy Vnn, ...
## ... the molecular integrals S, T, V, and G, and the number of AOs K
def compute_HFenergy(N, Vnn, S, T, V, G, K):
  """
  Calculate the electronic energy using the Hartree-Fock procedure,
  and then append to it the nuclear-nuclear repulsion energy.
  (Handout 5)
  """
  Hcore = T + V # (Eq. 1)
  D     = np.zeros((K,K)) # create an empty KxK density matrix;
  P     = np.zeros((K,K)) # create the KxK two-electron contribution

  X     = linalg.sqrtm(linalg.inv(S)) # generate transformation matrix (Eq. 2)

  count = 1 # instantiate iteration counter
  convergence = 1e-5 # set the HF energy convergence threshold

  ## begin the SCF iteration
  Eel = 0.0 # initial energy is 0.0
  for iteration in range(scf_max_iter+1):
    E_old = Eel # set E_old to energy from previous cycle, for comparison
  
    ## using the indices m and n in lieu of "mu" and "nu" ...
    ## ... calculate the two-electron contribution ...
    ## ... by contracting the density matrix with the two-electron integrals
    ## (Eq. 3)
    for m in range(K):
      for n in range(K):
        P[m,n] = 0.0
        for l in range(K):
          for s in range(K):
            P[m,n] += D[l,s] * (G[m,n,s,l] - 0.5 * G[m,l,s,n])
  
    F = Hcore + P             # Fock matrix = 1e + 2e contribution (Eq. 4)
    Fp = X @ F @ X            # transform Fock matrix to orthonormal basis (Eq. 5)
    e, Cp = linalg.eigh(Fp)   # diagonalize orthonomalized Fock matrix
    C = X @ Cp                # calculate the molecular orbitals (Eq. 6)
 
    ## form a new density matrix D from the molecular orbitals C 
    ## (Eq. 7)
    for m in range(K):
      for n in range(K):
        D[m,n] = 0.0
        for a in range(int(N/2)):
          D[m,n] += 2 * (C[m,a] * C[n,a])
    
    Eel = 0.0 # reset current iteration's electronic energy to 0 Hartrees

    ## calculate the electronic energy, an expectation value 
    ## (Eq. 8)
    for m in range(K):
      for n in range(K):
        Eel += 0.5 * D[n,m] * (Hcore[m,n] + F[m,n])
  
    print('Eel (iteration {:2d}): {:12.6f}'.format(count,Eel)) 
    if (np.fabs(Eel - E_old) < convergence) and (iteration > 0):
      break
    count += 1

  ## (Eq. 9, Etot = Eel + Vnn)
  print("\nEt = Eel + Enn".format(Eel, Vnn, Eel+Vnn, count))
  print("   = {:.6f} + {:.6f}".format(Eel, Vnn))
  print("   = {:.6f} Hartrees ({} iterations)\n".format(Eel+Vnn, count))
