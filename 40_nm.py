from Modules import *
from ChirpedContraDC_v2 import ChirpedContraDC

# 40 nm
wvl = [1520e-9, 1610e-9]
resolution = 100
N1 = 500
if True:
	N1 = 500
	N2 = 300

	periods = [328, 326, 324, 322, 320, 318, 316]
	dw = [0, 0, 0, 0, 0, 0, 0]

	d = ChirpedContraDC(resolution = resolution, N=N1, period = periods[0]*1e-9)
	d.w1 += dw[0]*1e-9
	d.w2 += dw[0]*1e-9

	for p, dw in zip(periods, dw):
		dev = ChirpedContraDC(resolution = resolution, N=N1, period = p*1e-9)
		dev.w1 += dw*1e-9
		dev.w2 += dw*1e-9

		d = d + dev

	d.wvl_range = [1520e-9, 1610e-9]
	d.simulate()
	d.displayResults()

	# d_0 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 328e-9)
	# d_0.w1 += 0
	# d_0.w2 += 0

	# d_1 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 326e-9)
	# d_1.w1 += 0
	# d_1.w2 += 0

	# d_2 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 324e-9)
	# d_2.w1 += 0
	# d_2.w2 += 0

	# d_3 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 322e-9)
	# d_3.w1 += 0
	# d_3.w2 += 0

	# d_4 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 320e-9)
	# d_4.w1 += 0
	# d_4.w2 += 0

	# d_5 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 318e-9)
	# d_5.w1 += 0
	# d_5.w2 += 0

	# d_6 = ChirpedContraDC(wvl_range=wvl, N=N1, resolution=resolution, period = 316e-9)
	# d_6.w1 += 0
	# d_6.w2 += 0

	# d = d_0 + d_1 + d_2 + d_3 + d_4 + d_5 + d_6

	# for device in [d_0, d_1, d_2, d_3, d_4, d_5, d_6, d]:
	# 	device.simulate()
	# 	if device is not d:
	# 		plt.plot(device.wavelength*1e9, device.drop, "--")
	# 	else:
	# 		plt.plot(device.wavelength*1e9, device.drop, "k-")

	# plt.show()
	# d.displayResults()
