## STO-3G (from EMSL Basis Set Exchange): 
##   exponent coefficient alpha 'a'
a = \
   (
    (
     (    0.16885540,    0.62391373,    3.42525091),    # H  1s
     (    0.00000000,    0.00000000,    0.0000000 )     # H  2s,2p
                                                   ),
    (
     (    0.31364979,    1.15892300,    6.36242139),    # He 1s
     (    0.00000000,    0.00000000,    0.0000000 )     # He 2s,2p
                                                   ),
    (    
     (    0.7946505,     2.9362007,    16.1195750 ),    # Li 1s
     (    0.0480887,     0.1478601,     0.6362897 )     # Li 2s,2p
                                                   ), 
    (
     (    1.4871927,     5.4951153,    30.1678710 ),    # Be 1s
     (    0.0993707,     0.3055389,     1.3148331 )     # Be 2s,2p
                                                   ),
    (
     (    2.4052670,     8.8873622,    48.7911130 ),    # B  1s
     (    0.1690618,     0.5198205,     2.2369561 )     # B  2s,2p
                                                   ),
    (
     (    3.5305122,    13.0450960,    71.6168370 ),    # C  1s
     (    0.2222899,     0.6834831,     2.9412494 )     # C  2s,2p
                                                   ),
    (
     (    4.8856602,    18.0523120,    99.1061690 ),    # N  1s
     (    0.2857144,     0.8784966,     3.7804559 )     # N  2s,2p
                                                   ),
    (
     (    6.4436083,    23.8088610,   130.7093200 ),    # O  1s
     (    0.3803890,     1.1695961,     5.0331513 )     # O  2s,2p
                                                   ),
    (
     (    8.2168207,    30.3608120,   166.6791300 ),    # F  1s
     (    0.4885885,     1.5022812,     6.4648032 )     # F  2s,2p
                                                   ),
    (
     (   10.2052970,    37.7081510,   207.0156100 ),    # Ne 1s
     (    0.6232293,     1.9162662,     8.2463151 )     # Ne 2s,2p
                                                   ),
## third row atoms, which now include 3s and 3p orbitals, can be filled in like so
    (
     (   12.3623880,    45.6785110,   250.7724300 ),    # Na 1s
     (    0.9099580,     2.7978819,    12.0401930 ),    # Na 2s,2p
     (    0.1641751,     0.4125649,     1.4787406 )     # Na 3s,3p
                                                   ),
## fill in remaining third row atoms, if so inclined
#    (
#     (          0.0,           0.0,           0.0 ),    # X 1s
#     (          0.0,           0.0,           0.0 ),    # X 2s,2p
#     (          0.0,           0.0,           0.0 )     # X 3s,3p
#                                                   ),
## fourth row atoms, which include 4s, 4p, and maybe 3d orbitals, can be filled in like so
    (
     (  38.03332899,   140.5315766,   771.5103681 ),    # K  1s
     (  3.960373165,   12.17710710,   52.40203979 ),    # K  2s,2p
     ( 0.3987446295,   1.018782663,   3.651583985 ),    # K  3s,3p
     (0.08214006743,  0.1860011465,  0.5039822505 ),    # K  4s,4p; no 3d
                                                   ),
    (
     (  42.10144179,   155.5630851,   854.0324951 ),    # Ca 1s
     (  4.501370797,   13.84053270,   59.56029944 ),    # Ca 2s,2p
     (  0.477707930,   1.220531941,   4.374706256 ),    # Ca 3s,3p
     ( 0.0742952070,  0.1682369410,  0.4558489757 ),    # Ca 4s,4p; no 3d
                                                   ),
    (
     (  46.42135516,   171.5249862,   941.6624250 ),    # Sc 1s
     (  5.076992278,   15.61041754,   67.17668771 ),    # Sc 2s,2p
     (  0.552930024,   1.433088313,   4.698159231 ),    # Sc 3s,3p
     ( 0.1028307363,  0.2328538976,  0.6309328384 ),    # Sc 4s,4p
     ( 0.0649300112,  0.1682861055,  0.5517000679 ),    # Sc 3d
                                                   )#, (include this comma if more atoms added)
## fill in remaining third row atoms, if so inclined
#    (
#     (          0.0,           0.0,           0.0 ),    # X 1s
#     (          0.0,           0.0,           0.0 ),    # X 2s,2p
#     (          0.0,           0.0,           0.0 )     # X 3s,3p
#     (          0.0,           0.0,           0.0 )     # X 4s,4p
#     (          0.0,           0.0,           0.0 )     # X 3d
#                                                   ),
   )

## STO-3G (from EMSL basis set exchange): 
##   contraction coefficient 'd' applies to all atoms
d = ((0.44463454, 0.53532814, 0.15432897),     # 1s
     (0.70011547, 0.39951283,-0.09996723),     # 2s
     (0.39195739, 0.60768372, 0.15591627),     # 2p
     (0.91667696, 0.21754360,-0.22776350),     # 3s
     (0.48464037, 0.57776647, 0.00495151),     # 3p
     (1.13103444, 0.01960641,-0.30884412),     # 4s
     (0.54989495, 0.57152276,-0.12154686),     # 4p
     (0.28657326, 0.65554736, 0.21976795))     # 3d

## dictionary: name --> atomic number
Z = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,'K':19,'Ca':20,'Sc':21}

## minimal basis orbital configuration, atoms 1 - 10
## ... as well as Na from row 3, and K, Ca, Sc from row 4
subshells = {
            'H':['1s'],
           'He':['1s'],
           'Li':['1s','2s','2p'],
           'Be':['1s','2s','2p'],
            'B':['1s','2s','2p'],
            'C':['1s','2s','2p'],
            'N':['1s','2s','2p'],
            'O':['1s','2s','2p'],
            'F':['1s','2s','2p'],
           'Ne':['1s','2s','2p'],
           'Na':['1s','2s','2p','3s','3p'],
            'K':['1s','2s','2p','3s','3p','4s','4p'],
           'Ca':['1s','2s','2p','3s','3p','4s','4p'],
           'Sc':['1s','2s','2p','3s','3p','4s','4p','3d']
           }

def build_sto3Gbasis(atoms, R):
  """
  This function depends on the atoms array, which lists strings of atomic symbols of the atoms of the input
  molecule, in the same order as the input .xyz file.
  This function additionally depends on the molecule's coordinates R (a nested array), where each element is
  an array holding the coordinates of each atom, also in the same order as atoms.
  """

  sto3Gbasis = []
  K = 0 # indexing the orbitals ==> matrix size
  for i, atom in enumerate(atoms):
    for subshell in subshells[atom]:
      if subshell == '1s':
        sto3Gbasis.append( 
                           {
                             'Z': Z[atom],                # atom name --> atomic number
                             'o': subshell,               # append the orbital-type string ('1s','2s',etc.)
                             'R': R[ i ],                 # get list [x,y,z] of atom coordinates
                             'l': 0,                      # s orbital ==> 0 angular momentum
                             'm': 0,
                             'n': 0,
                             'a': a[ (Z[atom]-1) ][0],    # append list of 1s orbital exponential factors
                             'd': d[0]                    # append list of 1s orbital contraction coefficients
                           }
                         )
        K += 1
      if subshell == '2s':
        sto3Gbasis.append( 
                           { 
                             'Z': Z[atom],
                             'o': subshell,
                             'R': R[ i ],
                             'l': 0,
                             'm': 0,
                             'n': 0,
                             'a': a[ (Z[atom]-1) ][1],   # append list of 2s orbital exponential factors
                             'd': d[1]                   # append list of 2s orbital contraction coefficients
                           }
                         )
        K += 1
      if subshell == '2p':
        sto3Gbasis.append( 
                           { 
                             'Z': Z[atom],
                             'o': '2px',
                             'R': R[ i ],
                             'l': 1,                     # 2px orbital has angular momentum in X direction
                             'm': 0,
                             'n': 0,                     
                             'a': a[ (Z[atom]-1) ][1],   # 2p orbital exponent = 2s orbital exponent
                             'd': d[2]                   # append list of 2p orbital contraction coefficients
                           }
                         )
        K += 1
        sto3Gbasis.append( 
                           { 
                             'Z': Z[atom],
                             'o': '2py',
                             'R': R[ i ],
                             'l': 0,
                             'm': 1,                     # 2py orbital has angular momentum in Y direction
                             'n': 0,
                             'a': a[ (Z[atom]-1) ][1],
                             'd': d[2]
                           }
                         )
        K += 1
        sto3Gbasis.append( 
                           { 
                             'Z': Z[atom],
                             'o': '2pz',
                             'R': R[ i ],
                             'l': 0,                     
                             'm': 0,
                             'n': 1,                     # 2pz orbital has angular momentum in Z direction
                             'a': a[ (Z[atom]-1) ][1],
                             'd': d[2]
                           }
                         )
        K += 1
      if subshell == '3s':
        sto3Gbasis.append( 
                           { 
                             'Z': Z[atom],
                             'o': subshell,
                             'R': R[ i ],
                             'l': 0,
                             'm': 0,
                             'n': 0,
                             'a': a[ (Z[atom]-1) ][2],   # append list of 3s orbital exponential factors
                             'd': d[3]                   # append list of 3s orbital contraction coefficients
                           }
                         )
        K += 1
      if subshell == '3p':
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3px',
                            'R': R[ i ],
                            'l': 1,                     # 3px orbital has angular momentum in X direction
                            'm': 0,
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][2],   # 3p orbital exponent = 3s orbital exponent
                            'd': d[4]                   # append list of 3p orbital contraction coefficients
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3py',
                            'R': R[ i ],
                            'l': 0,
                            'm': 1,                     # 3py orbital has angular momentum in Y direction
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][2],
                            'd': d[4]
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3pz',
                            'R': R[ i ],
                            'l': 0,
                            'm': 0,
                            'n': 1,                     # 3pz orbital has angular momentum in Z direction
                            'a': a[ (Z[atom]-1) ][2],
                            'd': d[4]
                          }
                        )
        K += 1
      if subshell == '4s':
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': subshell,
                            'R': R[ i ],
                            'l': 0,
                            'm': 0,
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][3],   # append list of 4s orbital exponential factors
                            'd': d[5]                   # append list of 4s orbital contraction coefficients
                          }
                        )
        K += 1
      if subshell == '4p':
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '4px',
                            'R': R[ i ],
                            'l': 1,                     # 4px orbital has angular momentum in X direction
                            'm': 0,
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][3],   # 4p orbital exponent = 3s orbital exponent
                            'd': d[6]                   # append list of 4p orbital contraction coefficients
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '4py',
                            'R': R[ i ],
                            'l': 0,
                            'm': 1,                     # 4py orbital has angular momentum in Y direction
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][3],
                            'd': d[6]
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '4pz',
                            'R': R[ i ],
                            'l': 0,
                            'm': 0,
                            'n': 1,                     # 4pz orbital has angular momentum in Z direction
                            'a': a[ (Z[atom]-1) ][3],
                            'd': d[6]
                          }
                        )
        K += 1
      if subshell == '3d':
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3dx^2',
                            'R': R[ i ],
                            'l': 2,                     # 3dx^2 orbital
                            'm': 0,
                            'n': 0,                     
                            'a': a[ (Z[atom]-1) ][4],   # 3d orbital exponent
                            'd': d[7]                   # append list of 3d orbital contraction coefficients
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3dy^2',
                            'R': R[ i ],
                            'l': 0,
                            'm': 2,                     # 3dy^2 orbital
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][4],   # 3d orbital exponent
                            'd': d[7]                   # append list of 3d orbital contraction coefficients
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3dz^2',
                            'R': R[ i ],
                            'l': 0,
                            'm': 0,
                            'n': 2,                     # 3pz^2 orbital
                            'a': a[ (Z[atom]-1) ][4],   # 3d orbital exponent  
                            'd': d[7]
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3dxy',
                            'R': R[ i ],
                            'l': 1,                     # 3dxy orbital has angular momentum in X direction
                            'm': 1,                     # 3dxy orbital has angular momentum in Y direction
                            'n': 0,
                            'a': a[ (Z[atom]-1) ][4],   # 3d orbital exponent
                            'd': d[7]
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3dxz',
                            'R': R[ i ],
                            'l': 1,                     # 3dxz orbital has angular momentum in X direction
                            'm': 0,
                            'n': 1,                     # 3dxz orbital has angular momentum in Z direction
                            'a': a[ (Z[atom]-1) ][4],   # 3d orbital exponent
                            'd': d[7]
                          }
                        )
        K += 1
        sto3Gbasis.append( 
                          { 
                            'Z': Z[atom],
                            'o': '3dyz',
                            'R': R[ i ],
                            'l': 0,
                            'm': 1,                     # 3dyz orbital has angular momentum in Y direction
                            'n': 1,                     # 3dyz orbital has angular momentum in Z direction
                            'a': a[ (Z[atom]-1) ][4],
                            'd': d[7]
                          }
                        )
        K += 1
  return sto3Gbasis, K

b = \
(  
  (
    (0.3552322122E+02, 0.6513143725E+01, 0.1822142904E+01, 0.6259552659E+00, 0.2430767471E+00, 0.1001124280E+00), # H 1s
    (0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000) # H 2s 2p
  ),
  (
    (0.6598456824E+02, 0.1209819836E+02, 0.3384639924E+01, 0.1162715163E+01, 0.4515163224E+00, 0.1859593559E+00), # He 1s
    (0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000) # He 2s 2p
  ), 
  (
    (0.1671758462E+03, 0.3065150840E+02, 0.8575187477E+01, 0.2945808337E+01, 0.1143943581E+01, 0.4711391391E+00), # Li 1s
    (0.6597563981E+01, 0.1305830092E+01, 0.4058510193E+00, 0.1561455158E+00, 0.6781410394E-01, 0.3108416550E-01) # Li 2s 2p
  ),
  (
    (0.3128704937E+03, 0.5736446253E+02, 0.1604850940E+02, 0.5513096119E+01, 0.2140896553E+01, 0.8817394283E+00), # Be 1s
    (0.1363324744E+02, 0.2698375464E+01, 0.8386530829E+00, 0.3226600698E+00, 0.1401314882E+00, 0.6423251387E-01) # Be 2s 2p
  ),
  (
    (0.5060118369E+03, 0.9277671639E+02, 0.2595558190E+02, 0.8916442908E+01, 0.3462515703E+01, 0.1426055179E+01), # B 1s
    (0.2107957992E+02, 0.4167680144E+01, 0.1303936977E+01, 0.5023060351E+00, 0.2177279869E+00, 0.1008207576E+00) # B 2s 2p
  ),
  (
    (0.7402124284E+03, 0.1359399821E+03, 0.3814355754E+02, 0.1313342945E+02, 0.5098374267E+01, 0.2091930658E+01), # C 1s
    (0.3258407300E+02, 0.6431233653E+01, 0.2018972601E+01, 0.7771703961E+00, 0.3374516381E+00, 0.1553961623E+00) # C 2s 2p
  ),
  (
    (0.1022424465E+04, 0.1873113696E+03, 0.5253118501E+02, 0.1805352149E+02, 0.7001154689E+01, 0.2873770360E+01), # N 1s
    (0.5184179599E+02, 0.1020012298E+02, 0.3191378854E+01, 0.1225979264E+01, 0.5324058812E+00, 0.2451192875E+00) # N 2s 2p
  ),
  (
    (0.1337088568E+04, 0.2457868878E+03, 0.6908823792E+02, 0.2371541705E+02, 0.9216665989E+01, 0.3784729567E+01), # O 1s
    (0.7690051861E+02, 0.1514839585E+02, 0.4730071263E+01, 0.1812878882E+01, 0.7873108675E+00, 0.3623119858E+00) # O 2s 2p
  ),
  (
    (0.1666791343E+04, 0.3052711900E+03, 0.8572257721E+02, 0.2949566840E+02, 0.1143943581E+02, 0.4711391391E+01), # F 1s
    (0.1057142857E+03, 0.2089535714E+02, 0.6533906250E+01, 0.2502417861E+01, 0.1086785714E+01, 0.5000000000E+00) # F 2s 2p
  ),
  (
    (0.2000000000E+04, 0.3660000000E+03, 0.1028000000E+03, 0.3540000000E+02, 0.1370000000E+02, 0.5630000000E+01), # Ne 1s
    (0.1350000000E+03, 0.2670000000E+02, 0.8350000000E+01, 0.3200000000E+01, 0.1380000000E+01, 0.6350000000E+00) # Ne 2s 2p
  )
)

c = (
  (0.9163596281E-02, 0.4936149294E-01, 0.1685383049E+00, 0.3705627997E+00, 0.4164915298E+00, 0.1303340841E+00), # 1s
  (-0.1325278809E-01, -0.4699171014E-01, -0.3378537151E-01, 0.2502417861E+00, 0.5951172526E+00, 0.2407061763E+00), # 2s
  (0.3759696623E-02, 0.3767936984E-01, 0.1738967435E+00, 0.4180364347E+00, 0.4258595477E+00, 0.1017082955E+00) # 2p
  )

## dictionary: name --> atomic number
Z6g = {'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10}

## minimal basis orbital configuration, atoms 1 - 10
## ... as well as Na from row 3, and K, Ca, Sc from row 4
subshells6g = {
            'H':['1s'],
           'He':['1s'],
           'Li':['1s','2s','2p'],
           'Be':['1s','2s','2p'],
            'B':['1s','2s','2p'],
            'C':['1s','2s','2p'],
            'N':['1s','2s','2p'],
            'O':['1s','2s','2p'],
            'F':['1s','2s','2p'],
           'Ne':['1s','2s','2p'],
           }

def build_sto6gbasis(atoms, R):
  """
  This function depends on the atoms array, which lists strings of atomic symbols of the atoms of the input
  molecule, in the same order as the input .xyz file.
  This function additionally depends on the molecule's coordinates R (a nested array), where each element is
  an array holding the coordinates of each atom, also in the same order as atoms.
  """

  sto6gbasis = []
  K = 0 # indexing the orbitals ==> matrix size
  for i, atom in enumerate(atoms):
    for subshell in subshells6g[atom]:
      if subshell == '1s':
        sto6gbasis.append( 
                           {
                             'Z': Z6g[atom],                # atom name --> atomic number
                             'o': subshell,               # append the orbital-type string ('1s','2s',etc.)
                             'R': R[ i ],                 # get list [x,y,z] of atom coordinates
                             'l': 0,                      # s orbital ==> 0 angular momentum
                             'm': 0,
                             'n': 0,
                             'a': b[ (Z6g[atom]-1) ][0],    # append list of 1s orbital exponential factors
                             'd': c[0]                    # append list of 1s orbital contraction coefficients
                           }
                         )
        K += 1
      if subshell == '2s':
        sto6gbasis.append( 
                           { 
                             'Z': Z6g[atom],
                             'o': subshell,
                             'R': R[ i ],
                             'l': 0,
                             'm': 0,
                             'n': 0,
                             'a': b[ (Z6g[atom]-1) ][1],   # append list of 2s orbital exponential factors
                             'd': c[1]                   # append list of 2s orbital contraction
                            }
                          )
        K += 1
      if subshell == '2p':
        sto6gbasis.append( 
                           { 
                             'Z': Z6g[atom],
                             'o': '2px',
                             'R': R[ i ],
                             'l': 1,                     # 2px orbital has angular momentum in X direction
                             'm': 0,
                             'n': 0,                     
                             'a': b[ (Z6g[atom]-1) ][1],   # 2p orbital exponent = 2s orbital exponent
                             'd': c[2]                   # append list of 2p orbital contraction coefficients
                           }
                         )
        K += 1
        sto6gbasis.append( 
                           { 
                             'Z': Z6g[atom],
                             'o': '2py',
                             'R': R[ i ],
                             'l': 0,
                             'm': 1,                     # 2py orbital has angular momentum in Y direction
                             'n': 0,
                             'a': b[ (Z6g[atom]-1) ][1],
                             'd': c[2]
                           }
                         )
        K += 1
        sto6gbasis.append( 
                           { 
                             'Z': Z6g[atom],
                             'o': '2pz',
                             'R': R[ i ],
                             'l': 0,                     
                             'm': 0,
                             'n': 1,                     # 2pz orbital has angular momentum in Z direction
                             'a': b[ (Z6g[atom]-1) ][1],
                             'd': c[2]
                            }
                          )
        K += 1

  return sto6gbasis, K
