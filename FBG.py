from Modules import *
from ChirpedContraDC_v2 import ChirpedContraDC

d = ChirpedContraDC(N=800, period=324e-9)

d.resolution = 300
d.N_seg = 50
d.apod_shape = "tanh"

d.wvl_range = [1520e-9, 1600e-9]



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


if False:
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


d.resolution = 200
d.N_seg = 50
d.N = 1000
d2 = copy.copy(d)
d2.period -= 2e-9
d2.w1 -= 2e-9
d2.w2 -= 4e-9

d3 = copy.copy(d2)
d3.period -= 2e-9
d3.w1 -= 2e-9
d3.w2 -= 2e-9

d4 = copy.copy(d3)
d4.period -= 2e-9
d4.w1 -= 0e-9
d4.w2 -= 0e-9

d = d + d2 + d3 + d4
d.simulate()
d.displayResults()




