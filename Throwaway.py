from Modules import *
from ChirpedContraDC_v2 import *

d = ChirpedContraDC(resolution=20)
d.period = 3.2600000000000003e-7
d.w1 = 5.58e-7
d.w2 = 4.38e-7

d.simulate()
