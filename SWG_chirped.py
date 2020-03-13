from Modules import *
from ChirpedContraDC_v3 import ChirpedContraDC

# d = ChirpedContraDC(resolution=200, N_seg=100)
# d.N = 1400
# d.wvl_range = [1500e-9, 1600e-9]
# d.apod_shape = "tanh"
# d.simulate()
# d.displayResults()
# d.getPerformance()


# d2 = copy.copy(d)
# d2.period = [320e-9, 321.5e-9]
# d2.period_profile = np.linspace(321.5e-9, 320e-9, d.N_seg)
# d2.w1_profile = d.w1*np.ones(d.N_seg)
# d2.w2_profile = d.w2*np.ones(d.N_seg)

# d2.simulate()
# # d2.displayResults()
# d2.getPerformance()

# d3 = copy.copy(d2)
# d3.period_profile = np.linspace(323e-9, 320e-9, d.N_seg)
# d3.simulate()
# # d3.displayResults()
# d3.getPerformance()


# plt.figure()
# plt.plot(d.wavelength*1e9, d.drop, label="0 V, BW = 8.1 nm")
# plt.plot(d.wavelength*1e9, d2.drop, label="4 V, BW = 12.1 nm")
# plt.plot(d.wavelength*1e9, d3.drop, label="?? 8 V, BW = 17.6 nm ??")
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Drop Response (dB)")
# plt.legend()
# plt.show()


d = ChirpedContraDC(resolution=100)
d.kappa = 15e3
d.N = 1000
d.simulate()
# d.displayResults()
plt.plot(d.wavelength*1e9, np.exp(d.drop/10))
plt.show() 