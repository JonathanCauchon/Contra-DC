from Modules import *
from ChirpedContraDC_v2 import *

self = ChirpedContraDC(target_wvl=[1550e-9, 1560e-9])
self.N_seg = 5
self.resolution=30
self.simulate()
# self.displayResults()



def chirpIsKnown(self):
	ID = str(self.N_seg) + "_"  \
	+str(int(self.target_wvl[0]*1e9)) \
		+ "_" + str(int(self.target_wvl[-1]*1e9))

	if os.path.exists("Database/Chirp_profiles/"+ID+".txt"):
		return True
	else:
		return False


def saveChirp(self):	
	ID = str(self.N_seg) + "_"  \
		+str(int(self.target_wvl[0]*1e9)) \
		+ "_" + str(int(self.target_wvl[-1]*1e9))

	with open("Database/Chirp_profiles/"+ID+".txt", "w") as file:
		np.savetxt(file, (self.period_profile, self.w1_profile, self.w2_profile))


def fetchChirp(self):
	ID = str(self.N_seg) + "_"  \
	+str(int(self.target_wvl[0]*1e9)) \
	+ "_" + str(int(self.target_wvl[-1]*1e9))

	self.period, self.w1, self.w2 = np.loadtxt("Database/Chirp_profiles/"+ID+".txt")
	return self.period, self.w1, self.w2

if chirpIsKnown(self):
	print("Known")
	p, w1, w2 = fetchChirp(self)
	print(p, w1, w2)
else:
	print("Unknown")
	saveChirp(self)
