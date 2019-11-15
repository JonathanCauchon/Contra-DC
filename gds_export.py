"""
Parameters to export
z: z position of period start
p: period length
w1, w2: width of segment
K: kappa of segment (prop. to corrugation width)
"""
from Modules import *
from ChirpedContraDC_v2 import *

def gdsExport(target_wvl, N1, N2, Nin, resolution, showFig=True, saveAs=None):
	# find bandwidths
	dummy = ChirpedContraDC(N=N1, resolution=100)
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
	wvls = np.linspace(min(target_wvl),max(target_wvl), Nin+2)
	middle_wvls = wvls[1:-1]
	middle_devices = []

	# for wvl in middle_wvls:
	if True:
		device = ChirpedContraDC(N=N2)
		device.period, device.w1, device.w2 = device.fetchParams(1550e-9)

		# for gds
		device.getApodProfile()
		l_seg = int(device.N/device.N_seg)
		K = np.repeat(device.apod_profile, l_seg)
		z = np.linspace(0, device.period*device.N, device.N)
		p = device.period*np.ones(device.N)
		w1 = device.w1*np.ones(device.N)
		w2 = device.w2*np.ones(device.N)

		print(K.size, z.size, p.size, w1.size, w2.size)


target_wvl = [1550e-9, 1560e-9]
N1=500
N2=300
Nin=1
resolution=10

gdsExport(target_wvl, N1, N2, Nin, resolution, showFig=True, saveAs=None)


	# 	middle_devices.append(device)

	# if target_wvl[-1] > target_wvl[0]:
	# 	d_middle = middle_devices[0]
	# 	for i in range(len(middle_devices)-1):
	# 		d_middle = d_middle + middle_devices[i+1]
	# else:
	# 	d_middle = middle_devices[-1]
	# 	for i in range(len(middle_devices)-1, 0, -1):
	# 		d_middle = d_middle + middle_devices[i-1]

	# if target_wvl[-1] > target_wvl[0]:
	# 	device = d_0 + d_middle + d_end
	# else:
	# 	device = d_0 + d_middle + d_end

	# device.wvl_range = [1500e-9, 1650e-9]
	# device.resolution = resolution
	# device.simulate()

	# subfig.set_title("N2 = " + str(N2)+", Nin = "+str(int(Nin)))
	# subfig.plot(device.wavelength*1e9, device.thru)
	# subfig.plot(device.wavelength*1e9, device.drop)
	# subfig.set_xlabel("Wavelength (nm)")
	# subfig.set_ylabel("Response (dB)")

	# if showFig:
	# 	plt.show()

	# if saveAs is not None:
	# 	plt.savefig(saveAs)