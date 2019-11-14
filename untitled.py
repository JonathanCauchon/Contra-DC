from Modules import *
from ChirpedContraDC_v2 import *

d = ChirpedContraDC(resolution=25)
# d.simulate()
# d.displayResults()

d2 = ChirpedContraDC(N=500)

d = d+d2
d.simulate()
d.displayResults()