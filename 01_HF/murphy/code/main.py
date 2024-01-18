import os.path # necessary for reading in previously genereated files

import time # to be used for calculating computation times
import math
import argparse # to be used for allowing user to specify input file
import numpy as np # mathematical, linear algebra lbirary
import hf, basis, oei, eri # import the other code files

## set print options of Numpy output for readability
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=6)
np.set_printoptions(linewidth=200)
np.set_printoptions(suppress=True)

## initiate timer for calculating computation time
start = time.time()

## dictionary: name --> atomic number
## rows 1 and 2 of periodic table; add to as desired if basis.py has basis set info
Z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10}

## let user specify .xyz input file when running this code
parser = argparse.ArgumentParser(description = 'Computes molecular integrals based on .xyz input.')
parser.add_argument('coords_file', metavar = 'coordinates.xyz', default = 'heh.xyz', type=str, help = '.xyz file')
args   = parser.parse_args()

fname     = args.coords_file[:-4]      # use input as output file name
inputfile = open(args.coords_file,'r') # open .xyz file as read-only

def constructNestedArray(data):
  """
  Build 2D nested list containing raw data from .xyz input file.
  """
  fileNestedList = [] # instantiate array to hold atoms and XYZ coordinates in useful matrix form

  n = int(data.readline())
  for line in data.readlines()[1:]:
      rowValues = []  # clear elements of line-by-line array after appending to "matrix"
      atom, x, y, z = line.split()
      fileNestedList.append([atom, float(x), float(y), float(z)])

  return n, fileNestedList

def constructAtomArray(nestedlist):
  """
  From 2D nested list based on .xyz, construct 1D array of atomic masses.
  """
  return [line[0] for line in nestedlist]

def trimMatrix(nestedlist):
  """
  Clip out first column of atom names, to give array of pure coordinates.
  """
  for column in nestedlist: del column[0]

n, R  = constructNestedArray(inputfile) # create nested array made up input file
atoms = constructAtomArray(R) # create array of atoms from nested array
print('Molecule contains '+str(n)+' atoms.')
print('Atoms: '+str(atoms))
trimMatrix(R) # create an array of pure coordinates by removing atom names from nested array

## build the STO-3G atomic orbital basis
orbitalbasis, K = basis.build_sto3Gbasis(atoms, R) # K = basis size ==> dimensions of matrix
# orbitalbasis, K = basis.build_sto6gbasis(atoms, R) # K = basis size ==> dimensions of matrix


## import or execute overlap integral evaluation
if os.path.exists('S.'+fname+'.npy') == True:
  S = np.load('S.'+fname+'.npy')
else:
  S = np.zeros((K,K))
  S = oei.buildS(orbitalbasis, S)
  print('Overlap integrals, (A|B), complete!')

  # save integrals as a numpy file that can be re-loaded
  Ssave = open('S.'+fname+'.npy','wb')
  np.save(Ssave,S)
  
  # print results in output file
  fout = open('out.S.'+str(fname)+'.txt','w')
  fout.write( str(S) )
  print('Results printed in out.S.'+fname+'.dat.')

## import or execute kinetic energy integral evaluation
if os.path.exists('T.'+fname+'.npy') == True:
  T = np.load('T.'+fname+'.npy')
else: 
  T = np.zeros((K,K)) 
  T = oei.buildT(orbitalbasis, T)
  print('\nKinetic energy integrals, (A|(-1/2*Lambda^2)|B), complete!')

  # save integrals as a numpy file that can be re-loaded
  Tsave = open('T.'+fname+'.npy','wb')
  np.save(Tsave,T)
  
  # print results in output file
  print('Results printed in out.T.'+fname+'.dat.')
  fout = open('out.T.'+fname+'.txt','w')
  fout.write( str(T) )

## import or execute nuclear attraction integral evaluation
if os.path.exists('V.'+fname+'.npy') == True:
  Vsum = np.load('V.'+fname+'.npy')
else:
  V = np.zeros((n,K,K)) 
  V = oei.buildV(orbitalbasis, V, R, Z, atoms)
  Vsum = np.zeros((K,K))
  for i in range(len(V)):
    Vsum += V[i]
  print('\nNuclear attraction integrals, (A|(1/r_iC)|B), complete!')

  # save integrals as a numpy file that can be re-loaded
  Vsave = open('V.'+fname+'.npy','wb')
  np.save(Vsave,Vsum)
  
  # print results in output file
  print('Results printed in out.V.'+fname+'.dat.')
  fout = open('out.V.'+fname+'.txt','w')
  fout.write( str(Vsum) )

  fout = open('out.H.'+fname+'.txt','w')
  fout.write( str(Vsum + T) )

## import or execute two-electron tensor evaluation
if os.path.exists('G.'+fname+'.npy') == True:
  G = np.load('G.'+fname+'.npy')
else: 
  print('\nBeginning two-electron integral evaluation, for '+str(K**4)+' matrix elements ...\n')
  G = np.zeros((K,K,K,K)) 
  G = eri.buildG(orbitalbasis,G,K)
  print('\nElectron repulsion integrals, (AB|CD), complete!')

  # save tensor as a numpy file that can be re-loaded
  Gsave = open('G.'+fname+'.npy','wb')
  np.save(Gsave,G)
  
  # print results in output file
  print('Results printed in out.G.'+fname+'.dat.')
  fout = open('out.G.'+fname+'.txt','w')
  fout.write( str(G) )

def electronCount(atoms):
  """
  Assuming a neutrally charged molecule, calculate the number of electrons,
  based on atomic numbers of molecule's constituent atoms.
  """
  N = 0
  for A in atoms:
    N += Z[A] 
  print('\nNumber of electrons: '+str(N))
  return N

def IJsq(RI,RJ):
  """
  Compute the square of the distance between two points.
  (Handout 4, Eq 6)
  """
  return sum( (RI[i]-RJ[i])**2 for i in (0,1,2) )

def nuclearRepulsion(atoms):
  """
  Calculate the nuclear-nuclear repulsion energy.
  (Handout 5, Eq. 9)
  """
  Vnn = 0.0
  for a, A in enumerate(atoms):
    for b, B in enumerate(atoms):
      if b > a:
        num  = Z[A] * Z[B]
        den  = math.sqrt( IJsq(R[a],R[b]) )
        Vnn += num / den
  return Vnn

N = electronCount(atoms) # count the number of electrons present
# N = 2 # for HeH+, for now 
Vnn = nuclearRepulsion(atoms) # calculate the nuclear repulsion energy

hf.compute_HFenergy(N,Vnn,S,T,Vsum,G, K) # call on HF procedure from hf.py
                                         # ... passing in integrals and other info

# calculate and display total computation time
end = time.time()
print("Calculation time: {:.3f} min. ({:.3f} sec.)".format((end-start)/60.,(end-start)))
