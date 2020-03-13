import sys
sys.path.append("../")
from Modules import *
from ChirpedContraDC_v3 import ChirpedContraDC

# d = ChirpedContraDC(a=10, N_seg=100, period=324e-9, N=1000)
# d.exportGdsInfo()


# # cdcs to be chirped
# Ns = np.arange(1000, 1300, 100)
# periods = [324e-9]

# for N in Ns:
# 	for p in periods:
# 		d = ChirpedContraDC(N=N, period=p, N_seg=100, apod_shape="tanh", resolution=100)
# 		d.exportGdsInfo()


# # Chirped CDCs
# d = ChirpedContraDC(w1 = [.56e-6, .57e-6],N=1000, period=324e-9, N_seg=100, apod_shape="tanh", resolution=100)
# d.exportGdsInfo(fileName="w1_Chirped_N_1000")


# d = ChirpedContraDC(w1 = [.56e-6, .57e-6],N=1200, period=324e-9, N_seg=100, apod_shape="tanh", resolution=100)
# d.exportGdsInfo(fileName="w1_Chirped_N_1200")

# d = ChirpedContraDC(w1 = [.56e-6, .57e-6], w2=[.44e-6, .45e-6], N=1200, period=324e-9, N_seg=100, apod_shape="tanh", resolution=100)
# d.exportGdsInfo(fileName="w1w2_Chirped_N_1200")










# No heaters
d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1200)
d.w1 = [.55e-6, .57e-6]
d.exportGdsInfo(fileName="no_heaters/w1_55_57")


d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1200)
d.w1 = [.56e-6, .57e-6]
d.exportGdsInfo(fileName="no_heaters/w1_56_57")


d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1200)
d.w2 = [.43e-6, .45e-6]
d.exportGdsInfo(fileName="no_heaters/w2_43_45")


d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1200)
d.w1 = [.43e-6, .44e-6]
d.exportGdsInfo(fileName="no_heaters/w2_43_44")


d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1200)
d.w1 = [.55e-6, .57e-6]
d.w2 = [.43e-6, .45e-6]
d.exportGdsInfo(fileName="no_heaters/w1_55_57_w2_43_45")



##

d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1000, resolution=100)
if True:
	d.w1 = [.55e-6, .57e-6]
	d.w2 = [.43e-6, .45e-6]
	d.N_seg /= 2
	d.getChirpProfile()
	d.N_seg *= 2
	d.N_seg = int(d.N_seg)

	w11 = .55e-6*np.ones(int(d.N_seg/2))
	w12 = d.w1_profile

	w22 = .45e-6*np.ones(int(d.N_seg/2))
	w21 = d.w2_profile


	d.w1_profile = np.append(w11, w12)
	d.w2_profile = np.append(w21, w22)


	d.period_profile = d.period*np.ones(d.N_seg)

d.exportGdsInfo(fileName="no_heaters/special_N_1000")


d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1100, resolution=100)
if True:
	d.w1 = [.55e-6, .57e-6]
	d.w2 = [.43e-6, .45e-6]
	d.N_seg /= 2
	d.getChirpProfile()
	d.N_seg *= 2
	d.N_seg = int(d.N_seg)

	w11 = .55e-6*np.ones(int(d.N_seg/2))
	w12 = d.w1_profile

	w22 = .45e-6*np.ones(int(d.N_seg/2))
	w21 = d.w2_profile


	d.w1_profile = np.append(w11, w12)
	d.w2_profile = np.append(w21, w22)


	d.period_profile = d.period*np.ones(d.N_seg)

d.exportGdsInfo(fileName="no_heaters/special_N_1100")


d = ChirpedContraDC(apod_shape="tanh", N_seg=100, period=320e-9, N=1200, resolution=100)
if True:
	d.w1 = [.55e-6, .57e-6]
	d.w2 = [.43e-6, .45e-6]
	d.N_seg /= 2
	d.getChirpProfile()
	d.N_seg *= 2
	d.N_seg = int(d.N_seg)

	w11 = .55e-6*np.ones(int(d.N_seg/2))
	w12 = d.w1_profile

	w22 = .45e-6*np.ones(int(d.N_seg/2))
	w21 = d.w2_profile


	d.w1_profile = np.append(w11, w12)
	d.w2_profile = np.append(w21, w22)


	d.period_profile = d.period*np.ones(d.N_seg)

d.exportGdsInfo(fileName="no_heaters/special_N_1200")
