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
import time

# t0 = time.time()
device = ChirpedContraDC(N=2000, apod_shape="tanh", period=316e-9)
# print(device.w1_profile.shape)
M = device.simulate().displayResults()
# print(time.time() - t0)

# print(M.shape)



# print(d_new.beta1_profile.shape)
# prit(d_new.beta1_profile.shape)
# t0 = time.time()
# d_new.simulate().getGroupDelay()
# t_new = time.time() - t0

# print(t_old, t_new)

# plt.figure()
# # plt.plot(d_old._wavelength, d_old.drop, "r", label="old")
# plt.plot(d_new._wavelength, d_new.drop, "k--", label="new")
# # plt.plot(d_old._wavelength, d_old.thru, "r")
# plt.plot(d_new._wavelength, d_new.thru, "k--")
# plt.legend()
# self = d_new
# # drop_phase = np.unwrap(np.angle(d_new.E_drop))
# thru_phase = np.unwrap(np.angle(d_new.E_thru))
# frequency = self.c/self.wavelength
# omega = 2*np.pi*frequency

# thru_group_delay = -np.diff(thru_phase)/np.diff(omega)
# thru_group_delay = np.append(thru_group_delay, thru_group_delay[-1])

# plt.plot(d_new._wavelength, d_new.group_delay, label="drop")
# plt.plot(d_new._wavelength, thru_group_delay, label="thru")

# plt.plot(d_new._wavelength, thru_group_delay+.44e-11, "k--", label="thru, shifted")
# plt.legend()
# plt.xlim((1550,1570))

# plt.plot(d_new._wavelength, drop_phase/np.pi, label="drop phase")
# plt.plot(d_new._wavelength, thru_phase/np.pi, label="thru phase")
# plt.ylabel("Phase ($\pi$)")
# plt.legend()
# plt.grid()
# plt.xlim((1550,1570))
# plt.xlabel("Wavelength (nm)")

# plt.twinx()
# plt.plot(d_new._wavelength, d_new.drop, "grey")
# plt.plot(d_new._wavelength, d_new.thru, "grey")
# plt.ylim((-60,5))
# plt.yticks([])
# plt.xticks([1550,1560,1570])
# plt.ylabel("Response (dB)")


# plt.show()



# plt.figure()
# # plt.plot(d_old._wavelength, d_old.group_delay, "r", label="old")
# plt.plot(d_new._wavelength, d_new.group_delay, label="new")
# plt.legend()

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