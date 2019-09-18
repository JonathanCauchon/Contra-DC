
from Modules import *

bestDevice = ChirpedContraDC(N=2500, a=13.3, resolution=500, wvl_range=[1530e-9,1600e-9], period=[316e-9,330e-9], DC=0.5)
bestDevice.simulate()
bestDevice.displayResults()