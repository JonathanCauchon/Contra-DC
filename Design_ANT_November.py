from Modules import *
from ChirpedContraDC_v2 import *

from Auto_design import autoDesign


# Intended designs
center_wvl = (1610+1530)*1e-9/2

# BWs = np.arange(15, 125, 5)*1e-9
# BWs = np.arange(15, 125, 5)*1e-9
BWs = np.arange(20, 75, 10)*1e-9

""" Fabricated devices
- 6 targeted BWs
- 3-4 N1-N2-Nin variations
- 3-4 wavelength offsets for experimental ref wvl uncertainty

- single stage forward, backward,  dual-stage same dir., alternated

"""

print(BWs)


# BWs = np.array([10e-9, 30e-9])
target_wvls = np.transpose([center_wvl + BWs/2, center_wvl - BWs/2])

folder_name = "Design_ANT_November"
if not os.path.isdir(folder_name):
	os.mkdir(folder_name)

N1 = 500
N2 = 300
resolution = 150


for target_wvl in target_wvls:

	file_name = "BW_" + str(abs(int(target_wvl[0]*1e9-target_wvl[-1]*1e9))) + "_nm.pdf"
	autoDesign(target_wvl, N1, N2, resolution, showFig=False, saveAs=folder_name+"/"+file_name)

