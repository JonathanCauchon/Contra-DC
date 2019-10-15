
from Modules import *
from ChirpedContraDC_v2 import *

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

#-----------------/ Hall of fame
HOF = [ChirpedContraDC(N=3800, a=5, w1=w1_, w2=w2_, period=p, resolution = 500, N_seg = 100, wvl_range=wr),
ChirpedContraDC(N=1000, resolution=100, wvl_range=[1500e-9,1600e-9], period=[312e-9,314e-9]),
ChirpedContraDC(a=1, N=1200, resolution=100, wvl_range=[1500e-9,1600e-9], period=[312e-9,316e-9]),
ChirpedContraDC(a=1, N=1600, resolution=100, wvl_range=[1500e-9,1600e-9], period=[312e-9,318e-9]),
ChirpedContraDC(a=1, N=1400, resolution=100, wvl_range=[1500e-9,1600e-9], period=[312e-9,318e-9]),
ChirpedContraDC(a=1, N=1300, resolution=100, wvl_range=[1500e-9,1600e-9], period=[312e-9,320e-9]),
ChirpedContraDC(a=1, N=1800, resolution=100, wvl_range=[1500e-9,1600e-9], period=[312e-9,322e-9])
]
#-----------------\


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


# -- October 8 2019 -- #

# d = ChirpedContraDC(resolution=100, period=p_, w1 = w1_, w2 = w2_)
# d.simulate()
# d.displayResults()
# d2 = ChirpedContraDC(a=1, resolution=100, N=1500, wvl_range = [1490e-9, 1600e-9], target_wvl = [1540e-9, 1555e-9])
# d2.simulate()
# d2.displayResults()

# HOF
HOF.append(ChirpedContraDC(N_seg=10, a=1,resolution=100, N=2100, wvl_range=[1490e-9,1640e-9],target_wvl=[1540e-9,1580e-9]))
HOF.append(ChirpedContraDC(N_seg=10, a=1,resolution=100, N=2400, wvl_range=[1490e-9,1640e-9],target_wvl=[1540e-9,1580e-9]))
HOF.append(ChirpedContraDC(N_seg=20, a=1,resolution=100, N=2145, wvl_range=[1490e-9,1640e-9],target_wvl=[1540e-9,1580e-9]))

N_seg = 100
# frac = 1/4
# # period
p = [312e-9, 324e-9]
# p1 = p[0]*np.ones(int(N_seg*frac))
# p2 = np.linspace(p[0], p[1], int(N_seg*(1-2*frac)))
# p2 = np.round(p2/2,9)*2
# p3 = p[1]*np.ones(int(N_seg*frac))

# p1=np.append(p1,p2)
# p1=np.append(p1,p3)
# p1=np.append(p1,p[1])
# print(p1)

def test(d):
	d.wvl_range = [1500e-9,1600e-9]
	d.a = 1
	d.resolution = 150
	d.stages = 2
	d.getApodProfile()
	d.getChirpProfile()
	d.chirpV2()
	d.getPropConstants(True)
	d.propagate(True)
	d.cascade()
	d.displayResults()


N_seg = 100
p = [314e-9, 322e-9]

d = ChirpedContraDC(period=p, N_seg=N_seg, a=1,resolution=150, N=1200, wvl_range=[1400e-9,1700e-9])
# 1200, 
# nums = np.linspace(1200, 2000, 50)
# for N in nums:
# 	d.N = N
# 	test(d)

# test(d)

MVP = [ChirpedContraDC(period=[312e-9, 324e-9], N_seg=50, a=1, N=2100)]
MVP.append(ChirpedContraDC(period=[316e-9, 322e-9], N_seg=50, a=1, N=1000))
MVP.append(ChirpedContraDC(period=[314e-9, 322e-9], N_seg=50, a=1, N=1200))

# test(MVP[-1])



# Broadening 
p = [318e-9, 318e-9]
N_seg = 100
devices = [ ChirpedContraDC(a=1,period=[318e-9, 318e-9], N_seg=N_seg, N=1000), \
			ChirpedContraDC(a=1,period=[316e-9, 320e-9], N_seg=N_seg, N=800), \
			ChirpedContraDC(a=1,period=[314e-9, 322e-9], N_seg=N_seg, N=1450), \
			ChirpedContraDC(a=1, period=[312e-9, 324e-9], N_seg=N_seg, N=2100), \
			ChirpedContraDC(a=1,period=[310e-9, 326e-9], N_seg=N_seg, N=2750) ]





saveFigs = True


thru = []
drop = []
# for d in devices:
# 	d.wvl_range = [1450e-9,1650e-9]
# 	# d.resolution = 10
# 	# d.stages = 1
# 	d.getApodProfile()
# 	d.getChirpProfile()
# 	d.chirpV2()
# 	d.getPropConstants(True)
# 	d.propagate(True)
# 	d.cascade()
# 	d.getPerformance()

# plt.figure()
# plt.axis((d.wavelength[0]*1e9, d.wavelength[-1]*1e9, -40, 5))
# for d in devices:
# 	plt.plot(d.wavelength*1e9, d.drop, label="BW: "+str(d.performance[1][1])+"nm")
# plt.legend()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Drop Response (dB)")
# if saveFigs:
# 	plt.savefig("Plots/Drop Spectral Broadening.pdf")

# plt.figure()
# for d in devices:
# 	plt.plot(d.wavelength*1e9, d.thru)
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Thru Response (dB)")
# if saveFigs:
# 	plt.savefig("Plots/Thru Spectral Broadening.pdf")

# Cascaded devices
# for d in devices:
# 	d.wvl_range = [1460e-9,1650e-9]
# 	# d.resolution = 100
# 	d.stages = 2
# 	d.getApodProfile()
# 	d.getChirpProfile()
# 	d.chirpV2()
# 	d.getPropConstants(True)
# 	d.propagate(True)
# 	d.cascade()
# 	d.getPerformance()

# plt.figure()
# plt.axis((d.wavelength[0]*1e9, d.wavelength[-1]*1e9, -75, 5))
# for d in devices:
# 	plt.plot(d.wavelength*1e9, d.drop, label="BW: "+str(d.performance[1][1])+"nm")
# plt.legend()
# plt.xlabel("Wavelength (nm)")
# plt.ylabel("Cascaded Drop Response (dB)")
# if saveFigs:
# 	plt.savefig("Cascaded Drop Spectral Broadening.pdf")



# plt.show()

# new results interface tests
d = ChirpedContraDC(resolution=10, period = [320e-9,322e-9])
d.simulate()
d.displayResults()





# print(d.apod_profile.size, d.apod_profile)

# d.N_seg = N_c
# d.getChirpProfile()
# # d.N_seg = N_seg
# p1 = d.period_profile[0]*np.ones(N_)
# p2 = d.period_profile[-1]*np.ones(N_)
# w11 = d.w1_profile[0]*np.ones(N_)
# # w12 = d.w1_profile[-1]*np.ones(N_)
# w21 = d.w2_profile[0]*np.ones(N_)
# # w22 = d.w2_profile[-1]*np.ones(N_)

# d.period_profile = np.append(p1, d.period_profile)
# d.period_profile = np.append(d.period_profile, p2)

# d.w1_profile = np.append(w11, d.w1_profile)
# while d.w1_profile.size < N_seg:
# 	d.w1_profile = np.append(d.w1_profile, d.w1_profile[-1])

# d.w2_profile = np.append(w21, d.w2_profile)
# while d.w2_profile.size < N_seg:
# 	d.w2_profile = np.append(d.w2_profile, d.w2_profile[-1])

# while d.period_profile.size < N_seg:
# 	d.period_profile = np.append(d.period_profile, p2[0])
# print(d.period_profile)
# d.getPropConstants(bar=True)
# d.propagate(bar=True)
# print(d.apod_profile.size)
# d.displayResults()


# Ns = np.arange(2000,2500,N_seg)
# As = np.arange(1,13,2)

# for N in Ns:
# 	d.N = N
# 	d.resolution = 200
# 	d.getPropConstants(bar=True)
# 	d.propagate(bar=True)
# 	print(d.N,d.N_seg)
# 	d.displayResults()
# w1 = [.56e-6,.565e-6]
# w2 = [.44e-6,.445e-6]
# d = ChirpedContraDC(a=1, N=1800, resolution=100,w1=w1,w2=w2, wvl_range=[1500e-9,1600e-9], period=[310e-9,330e-9])
# d = ChirpedContraDC(a=1, N=1800, period=320e-9, w1=w1,w2=w2,wvl_range=[1500e-9,1600e-9])
# d.simulate()
# d.displayResults()