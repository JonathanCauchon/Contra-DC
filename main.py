
from Modules import *
from ChirpedContraDC_v1 import *

# good design:
# bestDevice = ChirpedContraDC(N=2500, a=13.3, resolution=50, wvl_range=[1530e-9,1600e-9], period=[316e-9,330e-9], DC=0.5)

# device1 = ChirpedContraDC(N=1000, a=12, resolution=100, wvl_range=[1530e-9,1600e-9], period=320e-9, DC=0.5)

# device1.simulate()
# device1.displayResults()

# Same device, different index to the second decimal
# wg1_ = Waveguide(neff = 2.5516)
# wg2_ = Waveguide(neff = 2.3604)
# device2 = device1
# device2.wg1, device2.wg2 = wg1_, wg2_
# device2.simulate()
# device2.displayResults()


# width chirp
w1 = [.56e-6, .57e-6]
w2 = [.44e-6, .45e-6]
w1_ = [.57e-6, .55e-6]
w2_ = [.45e-6, .43e-6]
p  = [316e-9, 324e-9]
wr = [1500e-9,1600e-9]

d = ChirpedContraDC(w1=w1,w2=w2,resolution=50,a=5,period=p,N=1000, wvl_range=wr)
d.simulate()
d.displayResults()