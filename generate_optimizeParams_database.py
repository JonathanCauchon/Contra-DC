from Modules import *
from ChirpedContraDC_v2 import *

wavelengths = np.arange(1450e-9, 1650e-9, 0.25e-9)

cdc = ChirpedContraDC()
for i, wvl in enumerate(wavelengths):	
	period, w1, w2 = cdc.optimizeParams(wvl)
	with open("Database/Target_wavelengths.txt", "a+") as file:
		file.write(str(wvl)+" "+str(period)+" "+str(w1)+" "+str(w2)+"\n")
	print((i+1)/wavelengths.size*100)

# wavelength, period, w1, w2 = np.transpose(np.loadtxt("Database/Target_wavelengths.txt"))
