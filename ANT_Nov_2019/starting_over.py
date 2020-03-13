from Modules import *
from ChirpedContraDC_v3 import *




N = 1000
res = 100
lam1 = 1560e-9
lam2 = 1580e-9
K = 48e3
N_seg = 40



def pad(cdc, num):
	for i in range(num):
		cdc.period_profile = np.append(cdc.period_profile[0], cdc.period_profile)
		cdc.period_profile = np.append(cdc.period_profile, cdc.period_profile[-1])

		cdc.w1_profile = np.append(cdc.w1_profile[0], cdc.w1_profile)
		cdc.w1_profile = np.append(cdc.w1_profile, cdc.w1_profile[-1])

		cdc.w2_profile = np.append(cdc.w2_profile[0], cdc.w2_profile)
		cdc.w2_profile = np.append(cdc.w2_profile, cdc.w2_profile[-1])

		cdc.N_seg += 2

	return cdc





d = ChirpedContraDC(period=320e-9,resolution=res, N=N, kappa = K, N_seg = N_seg)
d.wvl_range = [1520e-9, 1610e-9]
d.apod_shape = "tanh"

# d.period_chirp_step = .1e-9
d.width_chirp_step = .1e-9
d.w1 = [.56e-6, .564e-6]
d.w2 = [.44e-6, .444e-6]

# tgt = np.linspace(lam1, lam2, d.N_seg)
# d.getChirpProfile()

# for i in range(d.N_seg):
# 	d.period_profile[i], d.w1_profile[i], d.w2_profile[i] = d.fetchParams(tgt[i])
d.getChirpProfile()
d = pad(d, 15)
# # d.simulate()
# # d.displayResults()

# for _ in range(1):
# 	for i in range(d.N_seg - 2):
# 		d.w1_profile[i+1] = (d.w1_profile[i] + d.w1_profile[i+2])/2
# 		d.w2_profile[i+1] = (d.w2_profile[i] + d.w2_profile[i+2])/2

d.beta = 5
d.alpha = 2
d.simulate()
d.displayResults()
# beta = np.arange(1,6)
# alpha = np.arange(2, 6)
# i=0
# for a in alpha:
# 	for b in beta:
# 		d.alpha = a
# 		d.beta = b
# 		# print(a, b)
# 		d.simulate(bar=False)
# 		plt.plot(d.wavelength, d.drop, label=str(a)+"_"+str(b))
# 		i +=1
# 		print(i)

# plt.legend()
# plt.show()


"""
a = 2
b = 3

"""

# z = np.arange(0, N_seg)
# beta = 3
# alpha = 2
# apod = 1/2 * (1 + np.tanh(beta*(1-2*abs(2*z/N_seg)**alpha)))
# apod = np.append(np.flip(apod[0:int(apod.size/2)]), apod[0:int(apod.size/2)])

# plt.plot(apod,"o")
# plt.show()