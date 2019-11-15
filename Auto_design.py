from Modules import *
from ChirpedContraDC_v2 import *

# d = ChirpedContraDC(a=20, resolution=300, wvl_range=[1400e-9, 1700e-9])

# d.target_wvl = [1540e-9, 1560e-9]
# # d.N = 2500
# d.N = 950
# d.N_seg = 5
# # d.getApodProfile()
# # d.getChirpProfile()
# # d.chirpV2()
# # d.getPropConstants(True)
# # d.propagate(True)
# d.stages = 1
# d.simulate()
# d.displayResults()



# for _ in range(2):
# 	d.period_profile = np.append(d.period_profile[0], d.period_profile)
# 	d.period_profile = np.append(d.period_profile, d.period_profile[-1])


# 	d.w1_profile = np.append(d.w1_profile[0], d.w1_profile)
# 	d.w1_profile = np.append( d.w1_profile, d.w1_profile[-1])

# 	d.w2_profile = np.append(d.w2_profile[0], d.w2_profile)
# 	d.w2_profile = np.append(d.w2_profile, d.w2_profile[-1])

# 	l_seg = int(d.N/d.N_seg)
# 	d.N_seg += 2
# 	d.N += 2*l_seg

# d.getApodProfile()
# d.getPropConstants(True)
# d.propagate(True)
# d.displayResults()


# --- Nov. 14 2019
# Tentative ultimate design methodology
# vini vidi vici

def plotApod():
	plt.plot(d3.apod_profile,"o")
	plt.show()


# resolution = 100
# w1_right = .57e-6
# w2_right = .45e-6
# w1_left = .55e-6
# w2_left = .43e-6

# d1 = ChirpedContraDC(N=1000, resolution=resolution, period=314e-9, w1=w1_right, w2=w2_right)
# d2 = ChirpedContraDC(N=300, resolution=resolution, period=318e-9)
# d3 = ChirpedContraDC(N=300, resolution=resolution, period=322e-9)
# d4 = ChirpedContraDC(N=1000, resolution=resolution, period=326e-9, w1=w1_left, w2=w2_left)
# # d2.simulate()

# # d3 = copy.copy(d1)
# # d3.apod_profile = np.append(d3.apod_profile, d2.apod_profile)
# # d3.period_profile = np.append(d3.period_profile, d2.period_profile)
# # d3.w1_profile = np.append(d3.w1_profile, d2.w1_profile)
# # d3.w2_profile = np.append(d3.w2_profile, d2.w2_profile)
# # d3.N += d2.N
# # d3.N_seg += d2.N_seg

# # d3.simulate()
# # d3.displayResults()
# d = d1 + d2 + d3 + d4
# d.wvl_range = [1500e-9, 1600e-9]
# d.simulate()
# d.displayResults()
# # plt.figure()
# # plt.plot(d1.apod_profile*.25)
# # plt.plot(d2.apod_profile*.5)
# # plt.plot(d3.apod_profile)
# # plt.show()

# Ultimate procudure !!!!!
N1 = 500
N2 = 350
numIntra = 8
target_wvl = [1610e-9, 1550e-9]


def autoDesign(N1, N2, target_wvl):
	# find bandwidths
	dummy = ChirpedContraDC(resolution=100)

	dummy.N = N1
	dummy.simulate()
	dummy.getPerformance()
	BW1 = dummy.performance[1][1]*1e-9

	dummy.N = N2
	dummy.simulate()
	dummy.getPerformance()
	BW2 = dummy.performance[1][1]*1e-9

	# extremities
	d_0 = ChirpedContraDC(N=N1)
	d_end = copy.copy(d_0)

	if target_wvl[-1] > target_wvl[0]:
		d_0.period, d_0.w1, d_0.w2 = dummy.optimizeParams(target_wvl[0]+BW1/2)
		d_end.period, d_end.w1, d_end.w2 = dummy.optimizeParams(target_wvl[-1]-BW1/2)
	else:
		d_0.period, d_0.w1, d_0.w2 = dummy.optimizeParams(target_wvl[0]-BW1/2)
		d_end.period, d_end.w1, d_end.w2 = dummy.optimizeParams(target_wvl[-1]+BW1/2)

	# center
	numIntra = abs(round(( target_wvl[-1]-BW1/2 - (target_wvl[0]+BW1/2) )/BW2))
	Nin = [numIntra-1, numIntra, numIntra+1]
	N2 = [N2-50, N2, N2+50]

	i, j = np.meshgrid(N2, Nin)
	fig, subfigs = plt.subplots(3 ,3, sharex="all", sharey="all", tight_layout=True, figsize=(9,6))
	fig.figsize= (10,6)

	for N2, Nin, subfig in zip(i.reshape(-1),j.reshape(-1), subfigs.reshape(-1)):

		wvls = np.linspace(min(target_wvl),max(target_wvl), Nin+2)
		middle_wvls = wvls[1:-1]
		middle_devices = []

		for wvl in middle_wvls:
			device = ChirpedContraDC(N=N2)
			device.period, device.w1, device.w2 = device.fetchParams(wvl)
			middle_devices.append(device)

		if target_wvl[-1] > target_wvl[0]:
			d_middle = middle_devices[0]
			for i in range(len(middle_devices)-1):
				d_middle = d_middle + middle_devices[i+1]
		else:
			d_middle = middle_devices[-1]
			for i in range(len(middle_devices)-1, 0, -1):
				d_middle = d_middle + middle_devices[i-1]

		if target_wvl[-1] > target_wvl[0]:
			device = d_0 + d_middle + d_end
		else:
			device = d_0 + d_middle + d_end

		device.wvl_range = [1500e-9, 1650e-9]
		device.resolution = 20
		device.simulate()

		subfig.set_title("N2 = " + str(i)+", Nin = "+str(j))
		subfig.plot(device.wavelength, device.thru)

	plt.show()



autoDesign(N1, N2, target_wvl)

# device = d_0 + d_end
# device.simulate()
# device.displayResults()










# dummy.target_wvl = [target_wvl[0], target_wvl[0]]
# print(dummy.period, dummy.w1, dummy.w2)

# print(dummy.period, dummy.w1, dummy.w2)



# d = ChirpedContraDC(wvl_range=[1500e-9, 1600e-9],resolution=50, period=308e-9, w1=.557e-6, w2=.437e-6)
# d.simulate()
# d.displayResults()




































