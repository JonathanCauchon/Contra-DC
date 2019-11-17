from Modules import *
from ChirpedContraDC_mexp import *

# d = ChirpedContraDC(N=100, kappa = 10_000, resolution=1)
# d.simulate()
# d.displayResults()
M = [[1, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1]]
A = sympy.Matrix(M)
print(A)
P, D = A.diagonalize()
print(P, D)
