from Modules import *
from ChirpedContraDC_v2 import *

d = ChirpedContraDC(a=20, resolution=300, wvl_range=[1400e-9, 1700e-9])

d.target_wvl = [1540e-9, 1560e-9]
# d.N = 2500
d.N = 950
d.N_seg = 5
# d.getApodProfile()
# d.getChirpProfile()
# d.chirpV2()
# d.getPropConstants(True)
# d.propagate(True)
d.stages = 1
d.simulate()
d.displayResults()



for _ in range(2):
	d.period_profile = np.append(d.period_profile[0], d.period_profile)
	d.period_profile = np.append(d.period_profile, d.period_profile[-1])


	d.w1_profile = np.append(d.w1_profile[0], d.w1_profile)
	d.w1_profile = np.append( d.w1_profile, d.w1_profile[-1])

	d.w2_profile = np.append(d.w2_profile[0], d.w2_profile)
	d.w2_profile = np.append(d.w2_profile, d.w2_profile[-1])

	l_seg = int(d.N/d.N_seg)
	d.N_seg += 2
	d.N += 2*l_seg

d.getApodProfile()
d.getPropConstants(True)
d.propagate(True)
d.displayResults()