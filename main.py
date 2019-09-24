
from Modules import *
from ChirpedContraDC_v1 import *

# good design:
# bestDevice = ChirpedContraDC(N=2500, a=13.3, resolution=50, wvl_range=[1530e-9,1600e-9], period=[316e-9,330e-9], DC=0.5)

# device1 = ChirpedContraDC(N=1000, a=12, resolution=100, wvl_range=[1530e-9,1600e-9], period=320e-9, DC=0.5)

# device1.simulate()
# device1.displayResults()

# Same device, different index to the second decimal
# wg1_ = Waveguide(neff = 2.5516)
# wg2_ = Waveguide(neff = 2.3604)
# device2 = device1
# device2.wg1, device2.wg2 = wg1_, wg2_
# device2.simulate()
# device2.displayResults()


# Chirp
w1 = [.55e-6, .57e-6]
w2 = [.45e-6, .44e-6]
w1_ = .55e-6
w2_ = .43e-6
p  = [320e-9, 324e-9]
p_ = 324e-9
wr = [1500e-9,1700e-9]
wr_ = [1530e-9,1600e-9]

# d = ChirpedContraDC(w1=.56e-6,w2=.44e-6,resolution=3,a=5,period=324e-9,N=1000, wvl_range=wr, N_seg = 50)
# d = ChirpedContraDC(a=5, w1=w1, w2=w2_, period=p_, resolution = 200, N_seg = 50, wvl_range=wr_)
# d.getApodProfile()
# d.getChirpProfile(plot=False)
# d.getPropConstants(plot=False)
# d.propagate()
# d.displayResults()

standard = ChirpedContraDC(period=330e-9, resolution=100, wvl_range=[1530e-9,1575e-9])
standard.simulate()
standard.getPerformance()
# standard.displayResults()
wvl_0 = standard.performance[0][1] # The basic reflection wavelength


ff = [1, 1, 1, 1]

if ff[0]:
	periods = np.arange(310e-9,330e-9,2e-9)
	lam = np.zeros((periods.size,1))
	cdc = standard
	cdc.wvl_range = [1500e-9,1600e-9]
	for i in range(periods.size):
		cdc.period = periods[i]
		cdc.simulate()
		cdc.getPerformance()
		lam[i] = cdc.performance[0][1]

	plt.figure()
	plt.plot(periods*1e9,lam,"o-")
	plt.xlabel("Grating period (nm)")
	plt.ylabel("Centre Reflection Wavelength (nm)")
	plt.savefig("Plots/centre_wvl_vs_period.pdf")
	# plt.show()


if ff[1]:
	w1s = np.arange(.55e-6,.57e-6,2e-9)
	lam = np.zeros((w1s.size,1))
	cdc = standard
	for i in range(w1s.size):
		cdc.w1 = w1s[i]
		cdc.simulate()
		cdc.getPerformance()
		lam[i] = cdc.performance[0][1]

	plt.figure()
	plt.plot(w1s*1e6,lam,"o-")
	plt.xlabel("Width of WG1 (um)")
	plt.ylabel("Centre Reflection Wavelength (nm)")
	plt.savefig("Plots/centre_wvl_vs_w1.pdf")
	# plt.show()


if ff[2]:
	w2s = np.arange(.43e-6,.45e-6,2e-9)
	lam = np.zeros((w2s.size,1))
	cdc = standard
	for i in range(w2s.size):
		cdc.w2 = w2s[i]
		cdc.simulate()
		cdc.getPerformance()
		lam[i] = cdc.performance[0][1]

	plt.figure()
	plt.plot(w2s*1e6,lam,"o-")
	plt.xlabel("Width of WG2 (um)")
	plt.ylabel("Centre Reflection Wavelength (nm)")
	plt.savefig("Plots/centre_wvl_vs_w2.pdf")
	# plt.show()


if ff[3]:
	w1s = np.arange(.55e-6,.57e-6,2e-9)
	w2s = np.arange(.43e-6,.45e-6,2e-9)
	lam = np.zeros((w2s.size,1))
	cdc = standard
	for i in range(w2s.size):
		cdc.w1 = w1s[i]
		cdc.w2 = w2s[i]
		cdc.simulate()
		cdc.getPerformance()
		lam[i] = cdc.performance[0][1]

	plt.figure()
	plt.plot((w2s-0.44e-6)*1e6,lam,"o-")
	plt.xlabel("Width Detuning (um)")
	plt.ylabel("Centre Reflection Wavelength (nm)")
	plt.savefig("Plots/centre_wvl_vs_w1w2.pdf")
	plt.show()