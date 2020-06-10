# import time

# resolution = 100
# N_seg = 100
# N = 1300
# period = [320e-9, 326e-9]
# # period_s

# w1, w2 = .56e-6, .45e-6

# Old model
# from ChirpedContraDC_v4 import *
# d_old = ChirpedContraDC(resolution=300, alpha=30)

# t0 = time.time()
# d_old.simulate(bar=True).getGroupDelay()
# t_old = time.time() - t0


# new model
from ChirpedContraDC_v5 import *
d_new = ChirpedContraDC()

t0 = time.time()
d_new.simulate().getGroupDelay().displayResults()
t_new = time.time() - t0

# print(t_old, t_new)

# plt.figure()
# # plt.plot(d_old._wavelength, d_old.drop, "r", label="old")
# plt.plot(d_new._wavelength, d_new.drop, "k--", label="new")
# # plt.plot(d_old._wavelength, d_old.thru, "r")
# plt.plot(d_new._wavelength, d_new.thru, "k--")
# plt.legend()

# # plt.figure()
# # # plt.plot(d_old._wavelength, d_old.group_delay, "r", label="old")
# # plt.plot(d_new._wavelength, d_new.group_delay, "k--", label="new")
# # plt.legend()

# plt.show()



# from ChirpedContraDC_v5 import *

# d1 = ChirpedContraDC(period=314e-9)
# d2 = ChirpedContraDC(period=320e-9)
# d3 = ChirpedContraDC(period=326e-9)

# device = d1 + d2 + d3
# device.simulate().displayResults()


""" Taking this even further

# Simulate WDM systems on the fly:

d1 = ChirpedContraDC(period=322e-9)
d2 = ChirpedContraDC(period=326e-9)
d3 = ChirpedContraDC(period=330e-9)

device = (3*d1 + 3*d2 + 3*d3)
device.simulate().displayResults()

"""