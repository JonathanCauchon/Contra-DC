from Modules import *

# m = np.loadtxt("indices.txt")


# w1 = 5.6e-7
# w2 = 4.4e-7
# n = 100
# wavelength = np.linspace(1.4e-6,1.7e-6,n)


# def findIt(w1,w2,lam):
# 	idx_w1 = np.where(m==w1,1,0)[:,0]
# 	idx_w2 = np.where(m==w2,1,0)[:,1]
# 	idx_lam = np.where(m==lam,1,0)[:,2]
# 	# print(idx_w1)
# 	# print(idx_w2) # for debugging
# 	# print(idx_lam)
# 	idx = np.where(idx_w1*idx_w2*idx_lam==1)
# 	m_ = m[idx]
# 	n1 = m[idx,3]
# 	n2 = m[idx,4]

# 	return n1, n2



# w2 = np.array([.43,.435,.44,.445,.45])*1e-6
# w1 = np.array([.55,.555,.56,.565,.57])*1e-6
# lam = np.array([1.4, 1.475, 1.55, 1.625, 1.7])*1e-6

# n1 = np.zeros((5,5,5))
# n2 = np.zeros((5,5,5))
# for i in range(w1.size):
# 	for j in range(w2.size):
# 		for k in range(lam.size):
# 			n1[i,j,k], n2[i,j,k] = findIt(round(w1[i],12),round(w2[j],12),round(lam[k],12))



# print(w1)
# print(w2)
# print(lam)

# # arr = np.ones((5,5,5))
# # xi = (0.56, .44, 1.55)
# # n1_, n2_ = scipy.interpolate.interpn(coords, n1, xi), scipy.interpolate.interpn(coords, n2, xi)
# # print(n1_, n2_)
def getNs(w1,w2,wvl):
	n1 = np.reshape(np.loadtxt("Database/neff/neff_1.txt"),(5,5,5))
	n2 = np.reshape(np.loadtxt("Database/neff/neff_2.txt"),(5,5,5))
	w1_w2_wvl = np.loadtxt("Database/neff/w1_w2_lambda.txt")
	n1_ = np.zeros((wvl.size,1))
	n2_ = np.zeros((wvl.size,1))
	for l in range(wvl.size):
		n1_[l] = scipy.interpolate.interpn(w1_w2_wvl, n1, [w1,w2,wvl[l]])
		n2_[l] = scipy.interpolate.interpn(w1_w2_wvl, n2, [w1,w2,wvl[l]])
	
	return n1_, n2_

ll = np.linspace(1400e-9,1700e-9,1000)
n1, n2 = getNs(.56e-6,.44e-6,ll)
print(n1.size)

# n1 = np.loadtxt("Database/neff/neff_1.txt")
# print(np.reshape(n1,(5,5,5)).shape)


