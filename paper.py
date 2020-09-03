from modules import *
from ChirpedContraDC_v7 import *


""" Figure 1: simulation """


d = ChirpedContraDC(N=1200,
					apod_shape="tanh",
					period=321e-9, 
					wvl_range=[1530e-9,1570e-9])
d.getApodProfile()
z = 1e6*np.linspace(0,d.N*d.period,d.N_seg)

plt.figure(figsize=(4,4))
plt.xlabel("z (a.u.)", fontsize=14)
plt.ylabel("$\kappa$ (mm$^{-1}$)", fontsize=14)
plt.xticks([])
d2 = ChirpedContraDC(N=1200,
					apod_shape="gaussian",
					period=315e-9, 
					wvl_range=[1530e-9,1570e-9])
d2.getApodProfile()
plt.plot(z, d2.apod_profile/1e3, "b", label="gaussian")
plt.plot(z, d.apod_profile/1e3, "r", label="tanh")

plt.legend()
plt.tight_layout()
plt.figure(figsize=(4,4))
plt.xlabel("Wavelength (a.u.)", fontsize=14)
plt.ylabel("Drop Response (dB)", fontsize=14)
d.simulate()
d2.simulate()
plt.plot(d._wavelength, d.drop, "r")
plt.plot(d._wavelength, d2.drop, "b")
plt.xticks([])
plt.tight_layout()

plt.show()
# fig, ax = plt.subplots(2,2, figsize=(8,8))

# # ax01 = ax[0,0].twinx()


# # Ts = [[350,350], [330,350], [320,350], [310,350], [300,350],[320,370],[335,385],[350,400], [370,400], [390,400], [400,400]]
# # Ts = [[300,310], [300,330], [300,360], [300,380], [300,400],[330,400],[360,400],[380,400], [390,400]]
# Ts = [[300,310], [310,320], [320,330], [330,340], [340,350]]
# Ts_ = [[320,330], [315,335], [310,340], [305,345], [300,350]]
# Ts_ = [[300,310], [300,320], [300,330], [300,340], [300,350]]

# colors = [[0,0,1], [.1,0,.9], [.2,0,.8], [.3,0,.7], [.4,0,.6], [.5,0,.5], [.6,0,.4], [.7,0,.3], [.8,0,.2], [.9,0,.1], [1,0,0]]
# # alpha = [1,.9,.8,.7,.6,.7,.8,.9,1]
# colors = [[0,0,1], [.25,0,.75], [.5,0,.5], [.75,0,.25], [1,0,0]]

# z = 1e6*np.linspace(0,d.N*d.period,d.N_seg)


# for T, T_, c in zip(Ts, Ts_, colors):
# 	d.T_profile = np.linspace(T[0],T[-1],d.N_seg)
# 	d.simulate()
# 	ax[0,0].plot(z, d.T_profile, color=c, alpha=1)
# 	ax[1,0].plot(d._wavelength, d.drop, color=c, alpha=1)

# 	d.T_profile = np.linspace(T_[0],T_[-1],d.N_seg)
# 	d.simulate()
# 	ax[0,1].plot(z, d.T_profile, color=c, alpha=1)
# 	ax[1,1].plot(d._wavelength, d.drop, color=c, alpha=1)

# # ax01.plot([0,0], [0,0],"k-", label="$\longleftarrow$")
# # ax01.plot(z, d.apod_profile/1e3, "k--", label="$\longrightarrow$")
# # ax01.set_ylabel("$\kappa$ (mm$^{-1}$)")


# ax[0,0].set_xlabel("z ($\mu$m)", fontsize=14)
# ax[0,0].set_ylabel("Temperature (K)", fontsize=14)
# # ax01.legend()

# ax[1,0].set_ylim((-25,1))
# ax[1,1].set_ylim((-25,1))
# ax[1,1].set_yticklabels([])
# ax[0,1].set_yticklabels([])
# ax[1,1].set_xlabel("Wavelength (nm)", fontsize=14)
# ax[0,1].set_xlabel("z ($\mu$m)", fontsize=14)
# ax[1,0].set_xlabel("Wavelength (nm)", fontsize=14)
# ax[1,0].set_ylabel("Drop Response (dB)", fontsize=14)


# plt.show()
