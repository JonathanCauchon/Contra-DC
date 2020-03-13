# Example of ChirpedContraDC_v3 class usage
# Many more optional properties inside class declaration

from Modules import *
from ChirpedContraDC_v4 import *


# 3 customizable linear chirps
w1 = [.55e-6, .57e-6]
w2 = [.43e-6, .45e-6]
w1 = .56e-6
w2 = .44e-6
p = [320e-9]
kappa = 30
# wvl range to plot
wr = [1500e-9, 1600e-9] 
N = 1400
apod_shape = "gaussian"

# Device creation, simulation and performance assessment
device = ChirpedContraDC(w1=w1, w2=w2, kappa=kappa*1e3, N=N, apod_shape=apod_shape, period=p, wvl_range=wr, resolution=50, N_seg=50)
# device.w_chirp_step = .1e-9
# device.period_chirp_step = .01e-9
device.simulate()
device.displayResults()




# d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1400, resolution=50)
# if True:
# 	d.w1 = [.55e-6, .57e-6]
# 	d.w2 = [.43e-6, .45e-6]
# 	d.N_seg /= 2
# 	d.N_seg = int(d.N_seg)
# 	d.getChirpProfile()
# 	d.N_seg *= 2
# 	d.N_seg = int(d.N_seg)

# 	w11 = .55e-6*np.ones(int(d.N_seg/2))
# 	w12 = d.w1_profile

# 	w22 = .45e-6*np.ones(int(d.N_seg/2))
# 	w21 = d.w2_profile


# 	d.w1_profile = np.append(w11, w12)
# 	d.w2_profile = np.append(w21, w22)


# 	d.period_profile = d.period*np.ones(d.N_seg)

# d.simulate()
# d.displayResults()