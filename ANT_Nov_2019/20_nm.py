from Modules import *
from ChirpedContraDC_v2 import *

N1 = 500
N2 = 250

resolution = 150
dw = 1e-9

# Option 1
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
if False:
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

	# for device in [d_0, d_1, d_2, d_end, d]:
	# 	device.simulate()
	# 	if device is not d:
	# 		plt.plot(device.wavelength*1e9, device.drop, "--")
	# 	else:
	# 		plt.plot(device.wavelength*1e9, device.drop, "k-")

	# plt.show()
	d.simulate()
	d.displayResults()

# Option 3
if False:
	d_0 = ChirpedContraDC(N=500, resolution=resolution, period = 322e-9)
	d_0.w1 += 2*dw
	d_0.w2 += 2*dw

	d_1 = ChirpedContraDC(N=500, resolution=resolution, period=320e-9)
	d_1.w1 += 3*dw
	d_1.w2 += 3*dw

	d_2 = ChirpedContraDC(N=500, resolution=resolution, period=320e-9)
	d_2.w1 -= 3*dw
	d_2.w2 -= 3*dw

	d_end = ChirpedContraDC(N=500, resolution=resolution, period=318e-9)
	d_end.w1 -= 2*dw
	d_end.w2 -= 2*dw

	d = d_0 + d_1 + d_2 + d_end

	for device in [d_0, d_1, d_2, d_end, d]:
		device.simulate()
		if device is not d:
			plt.plot(device.wavelength*1e9, device.drop, "--")
		else:
			plt.plot(device.wavelength*1e9, device.drop, "k-")

	plt.show()
	d.displayResults()



if True:
	res = 150
	N = 300
	d1 = ChirpedContraDC(N=1000, resolution = res, period = 326e-9)
	d1.w1 -= 3e-9
	d1.w2 -= 3e-9

	d2 = ChirpedContraDC(N=1000, resolution = res, period = 324e-9)

	d2.w1 += 2e-9
	d2.w2 += 2e-9

	dummy = ChirpedContraDC(N=100, resolution=res, period = 324e-9, kappa = 0)
	dummy.w1 -= 3e-9
	dummy.w2 -= 3e-9

	# d3 = ChirpedContraDC(N=700, resolution = res, period = 322e-9)
	# d3.w1 -= 4e-9
	# d3.w2 -= 4e-9

	# d4 = ChirpedContraDC(N=1000, resolution = res, period = 324e-9)
	# # d4.w1 -= 3e-9
	# # d4.w2 -= 3e-9

	# d5 = ChirpedContraDC(N=1000, resolution=res, period=322e-9)
	# d2.w1 += 3e-9
	# d2.w2 += 3e-9





	d = d1 + dummy + d2

	# d.apod_profile[700:1200] = d.apod_profile[700]
	wvl = [1560e-9, 1580e-9]
	d.wvl_range = [1560e-9, 1580e-9]
	# d1.wvl_range = [1550e-9, 1595e-9]
	# d2.wvl_range = [1550e-9, 1595e-9]
	# d3.wvl_range = [1550e-9, 1595e-9]
	# d4.wvl_range = [1550e-9, 1595e-9]
	# d5.wvl_range = [1550e-9, 1595e-9]
	# d.simulate()
	# print(d.apod_profile.size)
	# idx = []
	# for i in range(d.apod_profile.size):
	# 	if d.apod_profile[i] == d.apod_profile[i-1]:
	# 		idx.append(i)

	# print(idx)
	# np.delete(d.apod_profile, idx)
	# plt.plot(d.apod_profile)
	# plt.show()


	d.simulate()
	# d1.simulate()
	# d2.simulate()
	# d3.simulate()

	# plt.plot(d.wavelength, d1.drop, "--")
	# plt.plot(d.wavelength, d3.drop, "--")
	# plt.plot(d.wavelength, d2.drop, "--")
	# plt.plot(d.wavelength, d.thru, "k-")
	# plt.plot(d.wavelength, d.drop, "k-")
	# plt.show()
	d.displayResults()







































