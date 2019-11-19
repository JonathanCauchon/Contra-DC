from Modules import *
from ChirpedContraDC_v3 import ChirpedContraDC

d = ChirpedContraDC(resolution=200, N=1000, period=318e-9)
# d.simulate()
# d.displayResults()


apodization = np.loadtxt("From Vince/V2-Tanh/apodization.txt")[:, -1]
chirp = np.loadtxt("From Vince/V2-tanh/chirp.txt")[:, -1]
shape = np.loadtxt("From Vince/V2-tanh/shape.txt")
# plt.plot(apodization[:,-1])

# plt.figure()
# plt.plot(chirp[:,-1])

# plt.figure()
# plt.plot(shape)
# plt.show()

d = ChirpedContraDC(period=324e-9)

d.resolution = 150
d.N_seg = 60
d.apod_shape = "tanh"

d.alpha = 2
d.beta = 3

d.wvl_range = [1520e-9, 1600e-9]

d.kappa = 48e3
d.N = int(chirp.size/np.mean(chirp))
d.N = 3700



# Adjustments
d.N = 500
d.kappa = 48e3


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




d.w1 = [.57e-6, .55e-6]
d.w2 = [.45e-6, .43e-6]
d.N_seg /= 2
d.getChirpProfile()
d.N_seg *= 2
d.N_seg = int(d.N_seg)

w11 = .57e-6*np.ones(int(d.N_seg/2))
w12 = d.w1_profile

w22 = .43e-6*np.ones(int(d.N_seg/2))
w21 = d.w2_profile


d.w1_profile = np.append(w11, w12)
d.w2_profile = np.append(w21, w22)


d.period_profile = d.period*np.ones(d.N_seg)


d = pad(d, 5)



d.simulate()
d.displayResults()




