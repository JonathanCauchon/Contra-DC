from Modules import *
from ChirpedContraDC_v2 import ChirpedContraDC






# 60 nm
wvl = [1520e-9, 1610e-9]
resolution = 100


N1 = 300
if True:
	N1 = 500
	N2 = 300

	periods = [332, 330, 328, 326, 324, 322, 320, 318, 316, 314, 312]
	dw = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

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
	d_60 = d


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
	d_40 = d

	# 20 nm
	d_0 = ChirpedContraDC(N=500, resolution=resolution, period = 324e-9)
	d_0.w1 -= 2*dw
	d_0.w2 -= 2*dw

	d_1 = ChirpedContraDC(N=500, resolution=resolution, period=322e-9)
	d_1.w1 += 3*dw
	d_1.w2 += 3*dw

	d_2 = ChirpedContraDC(N=500, resolution=resolution, period=322e-9)
	d_2.w1 -= 3*dw
	d_2.w2 -= 3*dw

	d_end = ChirpedContraDC(N=500, resolution=resolution, period=320e-9)
	d_end.w1 += 2*dw
	d_end.w2 += 2*dw

	d = d_0 + d_1 + d_2 + d_end

	# for device in [d_0, d_1, d_2, d_end, d]:
	# 	device.simulate()
	# 	if device is not d:
	# 		plt.plot(device.wavelength*1e9, device.drop, "--")
	# 	else:
	# 		plt.plot(device.wavelength*1e9, device.drop, "k-")

	# plt.show()
	# d.displayResults()
	d_20 = d
	d_20.simulate()


	plt.figure()
	plt.plot(d_20.wavelength*1e9, d_20.drop)
	plt.plot(d_40.wavelength*1e9, d_40.drop)
	plt.plot(d_60.wavelength*1e9, d_60.drop)

	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Response (dB)")
	plt.show()







