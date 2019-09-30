
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
w1 = [.555e-6, .56e-6]
w2 = [.443e-6, .448e-6]
w1_ = .55e-6
w2_ = .43e-6
p  = [312e-9, 338e-9]
p_ = 324e-9
wr = [1500e-9,1700e-9]
wr_ = [1530e-9,1600e-9]

# good one:
# d = ChirpedContraDC(N=3800, a=5, w1=w1_, w2=w2_, period=p, resolution = 500, N_seg = 100, wvl_range=wr)
# d.simulate()
# d.displayResults()


# d = ChirpedContraDC(stages = 2, resolution = 50)
# d.simulate()
# d.displayResults()







# standard = ChirpedContraDC(period=324e-9, resolution=200, wvl_range=[1550e-9,1600e-9])
# standard.simulate()
# standard.getPerformance()
# standard.displayResults()
# wvl_0 = standard.performance[0][1] # The basic reflection wavelength


ff = [0, 0, 0, 0, 0]

if ff[0]:
	periods = np.arange(310e-9,330e-9,2e-9)
	lam = np.zeros((periods.size,1))
	cdc = ChirpedContraDC(period=324e-9, resolution=100, wvl_range=[1500e-9,1600e-9])
	cdc.wvl_range = [1500e-9,1620e-9]
	for i in range(periods.size):
		cdc.period = periods[i]
		cdc.simulate()
		cdc.getPerformance()
		lam[i] = cdc.performance[0][1]

	plt.figure()
	plt.plot((periods-324e-9)*1e9,lam,"o-")
	plt.xlabel("Period Detuning (nm)")
	plt.ylabel("Centre Reflection Wavelength (nm)")
	plt.savefig("Plots/centre_wvl_vs_period.pdf")
	# plt.show()

# lam_p = lam

if ff[1]:
	w1s = np.arange(.55e-6,.57e-6,2e-9)
	lam = np.zeros((w1s.size,1))
	cdc = ChirpedContraDC(period=324e-9, resolution=100, wvl_range=[1550e-9,1580e-9])
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
	cdc = ChirpedContraDC(period=324e-9, resolution=100, wvl_range=[1550e-9,1580e-9])
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
	w1s = np.arange(.55e-6,.571e-6,2e-9)
	w2s = np.arange(.43e-6,.451e-6,2e-9)
	dw = w2s-0.44e-6
	lam = np.zeros((w2s.size,1))
	cdc = ChirpedContraDC(period=324e-9, resolution=100, wvl_range=[1550e-9,1580e-9])
	for i in range(w2s.size):
		cdc.w1 = w1s[i]
		cdc.w2 = w2s[i]
		cdc.simulate()
		cdc.getPerformance()
		lam[i] = cdc.performance[0][1]

	plt.figure()
	plt.plot(dw*1e9,lam,"o-")
	plt.xlabel("Width Detuning (nm)")
	plt.ylabel("Centre Reflection Wavelength (nm)")
	plt.savefig("Plots/centre_wvl_vs_w1w2.pdf")
	plt.show()


# print(lam)
# print(lam_p)
# print(periods)
# print(dw)

if ff[4]:
	# Contruction of a custom w1w2 chirp profile
	w1 = [.54e-6, .57e-6]
	w2 = [.43e-6, .45e-6]
	p = []


def estimate_wvl(period, dw):
	periods = np.arange(310e-9,330e-9,2e-9)
	lam_p = 1e-9*np.array([1526.7, 1532.7, 1538.2, 1543.6, 1549.7, 1555.8, 1561.2, 1566.7, 1572.7, 1578.2, 1583.6])
	lam = 1e-9*np.array([1560.5, 1561.7, 1563. , 1564.4, 1565.6, 1566.8, 1568., 1569.2, 1570.5, 1571.7, 1572.9])
	d_w = np.array([-1.00000000e-08, -8.00000000e-09, -6.00000000e-09, -4.00000000e-09, -2.00000000e-09, -5.29395592e-23,  2.00000000e-09,  4.00000000e-09, 6.00000000e-09,  8.00000000e-09,  1.00000000e-08])
	dlam_dp, p_0 = np.polyfit(periods, lam_p, 1)
	dlam_dw, w_0 = np.polyfit(d_w, lam, 1)

	wvl = dlam_dp*period + p_0 + dlam_dw*dw
	return wvl

p = 316e-9
dw = -8e-9



# proof = ChirpedContraDC(resolution = 100, period = p, w1 = .56e-6+dw, w2 = .44e-6+dw)
# proof.simulate()
# proof.getPerformance()

# print(estimate_wvl(p,dw)*1e9)
# print(proof.performance[0][1])
# plt.plot(np.linspace(1,30,30), np.linspace(1530,1580,30),"o")
# plt.ylabel("Targeted ref. wavelength Range (nm)")
# plt.xlabel("Discrtized segments along grating")
# plt.show()