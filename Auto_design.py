from Modules import *
from ChirpedContraDC_v2 import *

def autoDesign(target_wvl, N1, N2, resolution, showFig=True, saveAs=None):
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

		device.wvl_range = [min(target_wvl)-15, max(target_wvl)+15]
		device.resolution = resolution
		device.simulate()

		subfig.set_title("N2 = " + str(N2)+", Nin = "+str(int(Nin)))
		subfig.plot(device.wavelength*1e9, device.thru)
		subfig.plot(device.wavelength*1e9, device.drop)
		subfig.set_xlabel("Wavelength (nm)")
		subfig.set_ylabel("Response (dB)")

	if showFig:
		plt.show()

	if saveAs is not None:
		plt.savefig(saveAs)




































