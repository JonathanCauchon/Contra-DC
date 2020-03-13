# Example of ChirpedContraDC_v3 class usage
# Many more optional properties inside class declaration

from Modules import *
from ChirpedContraDC_v4 import *


# 3 customizable linear chirps
# w1 = [.55e-6, .57e-6]
# w2 = [.43e-6, .45e-6]
w1 = .56e-6
w2 = .44e-6
p = [310e-9, 330e-9]
period_chirp_step = .1e-9

N_seg = round((p[-1]-p[0])/period_chirp_step) + 1

# p = 320e-9
kappa = 40e3
# wvl range to plot
wr = [1450e-9, 1650e-9] 
N = 10000
apod_shape = "gaussian"
a = 0
# apod_shape = "tanh"

# Device creation, simulation and performance assessment
device = ChirpedContraDC(a=a, w1=w1, w2=w2,  \
	kappa=kappa, N=N, apod_shape=apod_shape,  \
	period=p, wvl_range=wr, resolution=700, N_seg=N_seg)
# device.w_chirp_step = .1e-9
device.period_chirp_step = period_chirp_step

device.simulate()
# plt.plot(device.wavelength*1e9, device.drop)

# device.plotGroupDelay_v2()
# plt.figure()
# plt.plot(device.wavelength*1e9, device.group_delay)

dropPhase = np.unwrap(np.angle(device.E_drop))
frequency = device.c / device.wavelength
omega = frequency*2*np.pi
group_delay = -np.diff(dropPhase)/np.diff(omega)
group_delay = np.squeeze(group_delay, axis=0)
print(np.shape(group_delay))

group_delay = np.append(group_delay, group_delay[-1])
print(np.shape(group_delay))

group_delay = np.squeeze(group_delay)
dropPhase = np.squeeze(dropPhase)



plt.plot(device.wavelength*1e9, dropPhase)

plt.figure()
plt.plot(device.wavelength*1e9, group_delay)


device.displayResults()


# plt.figure()
# plt.plot(device.wavelength*1e9, device.drop)
# plt.title("Response")

# plt.figure()
# plt.plot(device.wavelength[:-1]*1e9, device.dropGroupDelay[0])
# plt.title("Drop group Delay")

# # plt.show()
# device.displayResults()

# print(np.shape(device.dropGroupDelay[0][]))

# d = ChirpedContraDC(apod_shape="tanh", N_seg=100,  \
# period=320e-9, N=1400, resolution=50)
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