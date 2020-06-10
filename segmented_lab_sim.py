from Modules import *
from ChirpedContraDC_v4 import *

def moving_average(array, num):
	for i in range(num, np.size(array)-num):
		array[i] = np.mean(array[i-num:i+num])

	return array



device = ChirpedContraDC(period=318e-9,resolution=150, N_seg=100, a=10, wvl_range=[1.5e-6, 1.6e-6])

# device.T_profile = np.linspace(300, 400, device.N_seg)

# fab_chirp = np.append(1.1*np.ones(10), np.linspace(1.1, 0.85, 20), np.linspace(.85, 0.95, 10))

def make_chirp(a=1.017, b=1, c=1.002):
	fab_chirp = np.append(a*np.ones(20), np.linspace(a, b, 60))
	fab_chirp = np.append(fab_chirp,  np.linspace(b, c, 20))

	# for i in range(3,47):
	# 	fab_chirp[i] = np.mean(fab_chirp[i-3:i+3])
	fab_chirp = moving_average(fab_chirp, 3)

	return fab_chirp


z = np.arange(0,device.N_seg)*device.l_seg


if False:
	device.w2_profile = device.w2*np.ones(device.N_seg)
	device.period_profile = device.period*np.ones(device.N_seg)
	device.w1_profile = device.w1*np.ones(device.N_seg)

	device.simulate()
	normal_drop = device.drop
	# device.displayResults()

	device.w1_profile = device.w1*make_chirp(1.017, .991, .994)
	device.simulate()
	chirped_drop = device.drop

	plt.figure()
	plt.plot(device._wavelength, normal_drop, label="No Chirp", linewidth=1.5)
	plt.plot(device._wavelength, chirped_drop, label="chirp", linewidth=1.5)
	plt.xlabel("Wavelength (nm)")
	plt.ylabel("Drop Response (dB)")
	plt.legend()

	plt.figure()
	plt.plot(z*1e6, device.w1_profile*1e9, linewidth=1.5)
	plt.xlabel("z (um)")
	plt.ylabel("w1 (nm)")
	plt.show()


# Create loss function

wvl = np.load("wvl.npy")
drop = np.load("no_chirp_spectrum.npy")
drop_ = np.load("chirp_spectrum.npy")


def compute_loss(wvl, drop, bw_factor=3, noise_floor=(-50,-40)):

	# spectrum stats
	drop_max = max(drop)
	drop_3dB = wvl[drop > drop_max - 3]
	ref_wvl = np.mean([drop_3dB[0], drop_3dB[-1]])
	bw = drop_3dB[-1] - drop_3dB[0]
	
	# idx_cropped = np.where((wvl > ref_wvl - bw) and (wvl < ref_wvl + bw), wvl, [])
	wvl_cropped = wvl[(wvl > ref_wvl - bw_factor*bw)]
	wvl_cropped = wvl_cropped[(wvl_cropped < ref_wvl + bw_factor*bw)]
	drop_cropped = drop[wvl > ref_wvl - bw_factor*bw]
	drop_cropped = drop_cropped[:np.size(wvl_cropped)]

	if False:
		plt.plot(wvl*1e9, drop, label="Complete spectrum")
		plt.plot(wvl_cropped*1e9, drop_cropped, label="Used for optimization")
		plt.legend()
		plt.show()

	# perfect spectrum based on these stats
	idx_left = np.where(wvl_cropped < ref_wvl - bw/2)
	idx_band = np.where(drop_cropped > drop_max - 3)
	idx_right = np.where(wvl_cropped > ref_wvl + bw/2)
	wvl_left = wvl_cropped[idx_left]
	wvl_band = wvl_cropped[idx_band]
	wvl_right = wvl_cropped[idx_right]

	ideal = np.zeros(np.size(wvl_cropped))
	ideal[idx_left] = np.linspace(noise_floor[0], noise_floor[1], np.size(idx_left))
	ideal[idx_band] = np.zeros(np.size(idx_band))
	ideal[idx_right] = np.linspace(noise_floor[1], noise_floor[0], np.size(idx_right))

	if False:
		plt.plot(wvl*1e9, drop, label="Experimental")
		# plt.plot(wvl_cropped*1e9, drop_cropped, label="Experimental")
		plt.plot(wvl_cropped*1e9, ideal, label="Ideal")
		plt.legend()
		plt.show()

	""" computing the loss function
		two different criteria apply:
			- in band: stick to ideal
			- out of band: be lower than ideal
	"""
	ddrop = np.zeros(np.size(ideal))
	for i in range(np.size(ideal)):
		

		if ideal[i] == 0: # in band
			ddrop[i] = abs(ideal[i] - drop_cropped[i])

		else: # out of band
			if drop_cropped[i] < ideal[i]: # sidelobe below ideal
				ddrop[i] = 0 # then no loss contributed
			else: # if sidelobe over ideal then there is loss
				ddrop[i] = abs(ideal[i] - drop_cropped[i])

	# plot loss vs wavelength
	if False:
		plt.figure()
		plt.plot(wvl*1e9, drop,label="Experimental")
		plt.plot(wvl_cropped*1e9, ideal, label="Ideal")
		plt.ylabel("Response (dB)")
		# plt.legend()
		plt.xlabel("Wavelength (nm)")
		plt.twinx()
		
		plt.plot(wvl_cropped*1e9, ddrop, "k", label="loss")
		plt.ylabel("Loss")
		plt.legend()

		plt.show()


	loss = sum(ddrop)/np.size(ideal)
	return loss

# Only for simulations
def get_temp(currents, N_seg=100):
	R = 30
	power = np.zeros(np.size(currents))
	power = R * currents**2
	dT = power*100/12e-3 # educated guess
	temp = 300 + dT
	# temp = moving_average(temp, 3)
	temp = np.interp(range(N_seg), np.linspace(0,N_seg-1, np.size(currents)), temp)
	temp = moving_average(temp, 3)
	return temp


# Gradient descent algorithm
currents = np.sqrt(.4e-3 - .02e-3*np.linspace(0,15,16))
# plt.plot(currents*1e3)
# plt.show()
# currents = 1e-3*np.array([15,15,15,15,15,15,15,15,15,15,15,15,15,15,9,9])
# print(get_temp(currents))
# plt.plot(get_temp(currents),"o-")
# plt.show()

# Just for simulations
def measure_spectrum():
	device.simulate()
	return device.wavelength, device.drop


def gradient_descent(currents, which_currents, device, lr=.5e-3, eps=0.2):
	wavelength, drop = measure_spectrum()

	current_0 = currents
	loss_0 = compute_loss(wavelength, drop)

	current_new = current_0
	current_new[which_currents] += lr
	wavelength, drop = measure_spectrum()
	loss_new = compute_loss(wavelength, drop)

	while abs(loss_0 - loss_new) > eps:



def get_gradient():
	pass


device = ChirpedContraDC(period=318e-9,resolution=150, N_seg=100, a=10, wvl_range=[1.5e-6, 1.6e-6])
device.w2_profile = device.w2*np.ones(device.N_seg)
device.period_profile = device.period*np.ones(device.N_seg)
device.w1_profile = device.w1*make_chirp(1.017, .991, .994)

device.simulate()
drop_fab = device.drop

device.T_profile = get_temp(currents)
device.simulate()
drop_temp = device.drop

plt.plot(device._wavelength, drop_fab, label="no heat")
plt.plot(device._wavelength, drop_temp, label="heat")
plt.legend()
plt.show()




