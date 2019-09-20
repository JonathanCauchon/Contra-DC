"""
This script creates output values of neff_1, neff_2 and w1_w2_wvl
used for 3d interpolation


"""

import cmath, math
import sys, os, time
import numpy as np
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import sys
import warnings


m = np.loadtxt("indices.txt")


def findIt(w1,w2,lam): # finds row with corresponding w1, w2 lambda, outputs n1, n2
	idx_w1 = np.where(m==w1,1,0)[:,0]
	idx_w2 = np.where(m==w2,1,0)[:,1]
	idx_lam = np.where(m==lam,1,0)[:,2]
	# print(idx_w1)
	# print(idx_w2) # for debugging
	# print(idx_lam)
	idx = np.where(idx_w1*idx_w2*idx_lam==1)
	m_ = m[idx]
	n1 = m[idx,3]
	n2 = m[idx,4]

	return n1, n2


# Values used in simulation
w2 = np.array([.43,.435,.44,.445,.45])*1e-6
w1 = np.array([.55,.555,.56,.565,.57])*1e-6
lam = np.array([1.4, 1.475, 1.55, 1.625, 1.7])*1e-6

n1 = np.zeros((5,5,5))
n2 = np.zeros((5,5,5))
for i in range(w1.size):
	for j in range(w2.size):
		for k in range(lam.size):
			n1[i,j,k], n2[i,j,k] = findIt(round(w1[i],12),round(w2[j],12),round(lam[k],12))

print(n1)
print("\n\n")
print(n2)
print("\n\n")
print(w1)
print(w2)
print(lam)

"""
You can then copy the printed arrays in a text file and use
those 3 text files for supermode index interpolation in 3d
:D
"""