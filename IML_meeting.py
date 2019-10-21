from Modules import *
from ChirpedContraDC_v2 import *

w1 = [.55e-6, .57e-6]
w2 = [.43e-6, .45e-6]
p = [312e-9, 320e-9]


d0 = ChirpedContraDC(period=p, w1=w1, w2=w2)
# d0.simulate()
# d0.displayResults()


p = [318e-9, 318e-9]
N_seg = 100
devices = [ ChirpedContraDC(a=1,period=[318e-9, 318e-9], N_seg=N_seg, N=1000), \
			ChirpedContraDC(a=1,period=[316e-9, 320e-9], N_seg=N_seg, N=800), \
			ChirpedContraDC(a=1,period=[314e-9, 322e-9], N_seg=N_seg, N=1450), \
			ChirpedContraDC(a=1, period=[312e-9, 324e-9], N_seg=N_seg, N=2100), \
			ChirpedContraDC(a=1,period=[310e-9, 326e-9], N_seg=N_seg, N=2750) ]




# if True:# for d in devices:
	
# 	d = devices[3]
# 	d.resolution = 100
# 	d.wvl_range = [1450e-9,1650e-9]
# 	d.getApodProfile()
# 	d.getChirpProfile()
# 	# d.chirpV2()
# 	d.getPropConstants(True)
# 	d.propagate(True)

# 	t1 = d.drop

# 	d.chirpV2()
# 	d.getPropConstants(True)
# 	d.propagate(True)
	
# 	t2 = d.drop

# plt.plot(d.wavelength*1e9, t1)
# plt.plot(d.wavelength*1e9, t2)
# plt.show()

d = ChirpedContraDC(resolution=100, N_seg=20, target_wvl=[1525e-9,1575e-9])
d.simulate()
d.displayResults()