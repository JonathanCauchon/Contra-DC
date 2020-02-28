from Modules import *
from ChirpedContraDC_v2 import *

N1 = 500
N2 = 250

resolution = 100
dw = 1e-9

if False:
	d_0 = ChirpedContraDC(N=500, resolution=resolution, period = 322e-9)
	d_0.w1 -= dw
	d_0.w2 -= dw

	d_1 = ChirpedContraDC(N=500, resolution=resolution, period=320e-9)

	d_end = ChirpedContraDC(N=500, resolution=resolution, period=318e-9)
	d_end.w1 += dw
	d_end.w2 += dw

	d = d_0 + d_1 + d_end

	for device in [d_0, d_1, d_end, d]:#, d_2, d_end, d]:
		device.simulate()
		if device is not d:
			plt.plot(device.wavelength*1e9, device.drop, "--")
		else:
			plt.plot(device.wavelength*1e9, device.drop, "k-")

	plt.show()
	d.displayResults()


# Option 2
if True:
	d_0 = ChirpedContraDC(N=500, resolution=resolution, period = 322e-9)
	d_0.w1 -= 2*dw
	d_0.w2 -= 2*dw

	d_1 = ChirpedContraDC(N=500, resolution=resolution, period=320e-9)
	d_1.w1 += 3*dw
	d_1.w2 += 3*dw

	d_2 = ChirpedContraDC(N=500, resolution=resolution, period=320e-9)
	d_2.w1 -= 3*dw
	d_2.w2 -= 3*dw

	d_end = ChirpedContraDC(N=500, resolution=resolution, period=318e-9)
	d_end.w1 += 2*dw
	d_end.w2 += 2*dw

	d = d_0 + d_1 + d_2 + d_end

	for device in [d_0, d_1, d_2, d_end, d]:
		device.simulate()
		if device is not d:
			plt.plot(device.wavelength*1e9, device.drop, "--")
		else:
			plt.plot(device.wavelength*1e9, device.drop, "k-")

	plt.show()
	d.displayResults()


# 30 nm
if False:
	N1 = 500
	N2 = 300

	d_0 = ChirpedContraDC(N=N1, resolution=resolution, period = 322e-9)
	d_0.w1 += 0
	d_0.w2 += 0


	d_end = ChirpedContraDC(N=N1, resolution=resolution, period = 322e-9)
	d_end.w1 += 0
	d_end.w2 += 0

	d = d_0 + d_end
	d.simulate()
	d.displayResults()




