from modules import *
from ChirpedContraDC_v5 import *

device = ChirpedContraDC(resolution=500, 
						N_seg = 100)

import time

t0 = time.time()
device.simulate()
print(time.time() - t0)
