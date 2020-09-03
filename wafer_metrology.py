from modules import *
from ChirpedContraDC_v6 import *

N = 1000
periods = 1e-9*np.arange(312,336.1,4)
# periods = periods[[0,2,4,1,3,5]]
kappa = 15e3
resolution=500
wvl = [1525e-9, 1610e-9]

d = ChirpedContraDC(w_chirp_step=1e-15,  kappa=kappa, N=N, period=periods[0], resolution=resolution, wvl_range=wvl)

# d_bends = ChirpedContraDC(alpha=5000, kappa=0, N=1000, period=320e-9, resolution=resolution, wvl_range=wvl)
# d_void = ChirpedContraDC(alpha=0, kappa=0, N=1000, period=320e-9, resolution=resolution, wvl_range=wvl)

for period in periods[1:]:
	d_next = ChirpedContraDC(kappa=kappa, N=N, period=period, resolution=resolution, wvl_range=wvl)
	d = d + d_next

print(d.length*1e6)
d.simulate().displayResults()
# dd = d.drop

# d2 = copy.copy(d)
# d2.w_chirp_step = 1e-15
# d.w1_profile = None
# d.w1 = [.56e-6, .565e-6]
# d.simulate().displayResults()

# plt.plot(d._wavelength, dd, "k")
# plt.plot(d._wavelength, d.drop, "r--")
# plt.show()


# d = ChirpedContraDC()
# print(d.length)
# d.simulate().displayResults()
# d2 = ChirpedContraDC(alpha=200, kappa=20e3, N=1000, period=324e-9)
# d3 = d + d2
# d3.simulate().displayResults()