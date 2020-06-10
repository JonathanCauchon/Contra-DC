# Example of ChirpedContraDC_v4 class usage
# Many more optional properties inside class declaration

from Modules import *
from ChirpedContraDC_v3 import *


# grating parameters
w1 = .56e-6 # waveguide 1 width
w2 = .44e-6 # waveguide 2 width
period = 318e-9 # grating period
N = 1000 # number of grating periods

# simulation parameters
wr = [1530e-9, 1565e-9] # wavelength range to plot
res = 50 # number of wavelength points

# Device creation, simulation and performance assessment
device = ChirpedContraDC(w1=w1, w2=w2, N=N, period=period,\
						wvl_range=wr, resolution=res)
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