"""
	ChripedContraDC.py
	Chirped contra-directional coupler model
	Chirp your CDC, engineer your response
	(Or let a computer engineer it for you)
	
	Based on Matlab model by Jonathan St-Yves
	as well as Python model by Mustafa Hammood

	Jonathan Cauchon, September 2019

	
	Class ChirpedContraDC:
		Methods:
			- switchTop, swapCols, swapRows: basic matrix operations
			- printProgressBar: show progress of computation
			- 

	-- v2 novelties --
	v2 incorporates the target_wvl property, which is a 2-element list. When specified in 
	intantiation, it triggers a method called optimizChirp that will find the best chirp 
	profile combination for period, w1 and w2 in order to cover the requested reflection
	wavelength range. The number of total periods then has to be optimized to get a nice 
	response.

"""

"""   Notes
- Experimental vs simulated:
	Center wavelength: experimental is 8 nm higher than simulated center wvl

"""

from Modules import *

def clc():
	print ("\n"*10)


class ChirpedContraDC():
	def __init__(self, N = 1000, period = 322e-9, DC = 0.5, a = 12, kappa = 48000, T = 300, \
		resolution = 300, N_seg = 50, wvl_range = [1530e-9,1580e-9], central_wvl = 1550e-9, \
		alpha = 10, stages = 1, w1 = .56e-6, w2 = .44e-6, target_wvl = None):

		# Class attributes
		self.N           =  N           #  int    Number of grating periods      [-]
		self.period      =  period      #  float  Period of the grating          [m]
		self.a           =  a           #  int    Apodization Gaussian constant  [-]
		self.kappa       =  kappa       #  float  Maximum coupling power         [m^-1]
		self.T           =  T           #  float  Device temperature             [K]
		self.resolution  =  resolution  #  int    Nb. of freq. points computed   [-]
		self.N_seg       =  N_seg       #  int    Nb. of apod. & chirp segments  [-]
		self.alpha       =  alpha       #  float  Propagation loss grating       [dB/cm]
		self.stages      =  stages      #  float  Number of cascaded devices     [-]
		self.wvl_range   =  wvl_range   #  list   Start and end wavelengths      [m]
		self.w1          =  w1          #  float  Width of waveguide 1           [m]
		self.w2          =  w2          #  float  Width of waveguide 2           [m]
		self.target_wvl  =  target_wvl  # list    Targeted reflection wavelength range [m]
		# Note that gap is set to 100 nm

		# Constants
		self._antiRefCoeff = 0.01
		self.period_chirp_step = 2e-9 # To comply with GDS resolution
		self.w_chirp_step = 1e-9


		# Protecting the model against user-induced inconsistancies
		# Gives warnings, errors, makes corrections

		# Check if N is a multiple of N_seg
		# if self.N%self.N_seg:
		# 	print("Number of periods (N) should be an integer multiple of the number of segments (N_seg).")
		# 	self.N_seg = 50
		# 	print("Number of segments was changed to "+str(self.N_seg)+".")


	# Property functions: changing one property automatically affects others
	@ property
	def wavelength(self):
		return np.linspace(self.wvl_range[0], self.wvl_range[1], self.resolution)

	@ property
	def c(self):
		return 299792458


	# linear algebra numpy manipulation functions
	def switchTop(self, P):
		P_FF = np.asarray([[P[0][0],P[0][1]],[P[1][0],P[1][1]]])
		P_FG = np.asarray([[P[0][2],P[0][3]],[P[1][2],P[1][3]]])
		P_GF = np.asarray([[P[2][0],P[2][1]],[P[3][0],P[3][1]]])
		P_GG = np.asarray([[P[2][2],P[2][3]],[P[3][2],P[3][3]]])

		H1 = P_FF-np.matmul(np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1)),P_GF)
		H2 = np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1))
		H3 = np.matmul(-np.linalg.matrix_power(P_GG,-1),P_GF)
		H4 = np.linalg.matrix_power(P_GG,-1)
		H = np.vstack((np.hstack((H1,H2)),np.hstack((H3,H4))))

		return H

	# Swap columns of a given array
	def swap_cols(self, arr, frm, to):
		arr[:,[frm, to]] = arr[:,[to, frm]]
		return arr

	# Swap rows of a given array
	def swap_rows(self, arr, frm, to):
		arr[[frm, to],:] = arr[[to, frm],:]
		return arr
	    

	# Print iterations progress
	def printProgressBar (self, iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
		# Print New Line on Complete
		if iteration == total: 
		    print()


	# This performs a 3d interpolation to estimate effective indices
	def getPropConstants(self, bar, plot=False):
		
		T0 = 300
		dneffdT = 1.87E-04      #[/K] assuming dneff/dn=1 (well confined mode)
		neffThermal = dneffdT*(self.T-T0)

		# Import simulation results to be used for interpolation
		n1 = np.reshape(np.loadtxt("Database/neff/neff_1.txt"),(5,5,5))
		n2 = np.reshape(np.loadtxt("Database/neff/neff_2.txt"),(5,5,5))
		w1_w2_wvl = np.loadtxt("Database/neff/w1_w2_lambda.txt")

		# Estimated values
		self.n1_profile = np.zeros((self.resolution, self.N_seg))
		self.n2_profile = np.zeros((self.resolution, self.N_seg))
		self.beta1_profile = np.zeros((self.resolution, self.N_seg))
		self.beta2_profile = np.zeros((self.resolution, self.N_seg))

		if bar:
			progressbar_width = self.resolution
			clc()
			print("Calculating propagation constants...")		
			self.printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

		for i in range(self.resolution): # i=lambda, j=z
			if bar: 
				clc()
				print("Calculating propagation constants...")
				self.printProgressBar(i + 1, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

			for j in range(self.N_seg):
				self.n1_profile [i,j] = neffThermal + scipy.interpolate.interpn(w1_w2_wvl, n1, [self.w1_profile[j],self.w2_profile[j],self.wavelength[i]])
				self.n2_profile [i,j] = neffThermal + scipy.interpolate.interpn(w1_w2_wvl, n2, [self.w1_profile[j],self.w2_profile[j],self.wavelength[i]])

			self.beta1_profile [i,:] = 2*math.pi / self.wavelength [i] * self.n1_profile [i,:]
			self.beta2_profile [i,:] = 2*math.pi / self.wavelength [i] * self.n2_profile [i,:]


		if plot:
			p1n1 = self.n1_profile[0,:]
			p2n1 = self.n1_profile[round(self.resolution/2),:]
			p3n1 = self.n1_profile[-1,:]

			p1n2 = self.n2_profile[0,:]
			p2n2 = self.n2_profile[round(self.resolution/2),:]
			p3n2 = self.n2_profile[-1,:]

			plt.figure()
			plt.plot(range(self.N_seg),p1n1,"b-",label="n1, "+str(self.wavelength[0]))
			plt.plot(range(self.N_seg),p2n1,"b--",label="n1, "+str(round(self.wavelength[round(self.resolution/2)],8)))
			plt.plot(range(self.N_seg),p3n1,"b-.",label="n1, "+str(self.wavelength[-1]))

			plt.plot(range(self.N_seg),p1n2,"r-",label="n2, "+str(self.wavelength[0]))
			plt.plot(range(self.N_seg),p2n2,"r--",label="n2, "+str(round(self.wavelength[round(self.resolution/2)],8)))
			plt.plot(range(self.N_seg),p3n2,"r-.",label="n2, "+str(self.wavelength[-1]))

			plt.legend()
			plt.xlabel("Segment number")
			plt.ylabel("Supermode Effective Indices")
			plt.show()

			clc()




	def getApodProfile(self):
		ApoFunc=np.exp(-np.linspace(0,1,num=1000)**2)     #Function used for apodization (window function)
		mirror = False                #makes the apodization function symetrical

		l_seg = self.N*np.mean(self.period)/self.N_seg
		n_apodization=np.arange(self.N_seg)+0.5
		zaxis = (np.arange(self.N_seg))*l_seg

		if  self.a != 0:
		    kappa_apodG = np.exp(-self.a*((n_apodization)-0.5*self.N_seg)**2/self.N_seg**2)
		    ApoFunc = kappa_apodG

		profile = (ApoFunc-min(ApoFunc))/(max(ApoFunc)-(min(ApoFunc))) # normalizes the profile

		n_profile = np.linspace(0,self.N_seg,profile.size)
		profile = np.interp(n_apodization, n_profile, profile)
		    
		kappaMin = self.kappa*profile[0]
		kappaMax = self.kappa
		kappa_apod=kappaMin+(kappaMax-kappaMin)*profile

		self.apod_profile = kappa_apod

	# --------------/
	# Section relative to chirp and chirp optimization


	# This creates a regression to estimate the reflection wavelength
	# (Only used to get parameters in optimizeParams)
	def estimate_wvl(period, dw):
	
		periods = np.arange(310e-9,330e-9,2e-9)
		lam_p = 1e-9*np.array([1526.7, 1532.7, 1538.2, 1543.6, 1549.7, 1555.8, 1561.2, 1566.7, 1572.7, 1578.2, 1583.6])
		lam = 1e-9*np.array([1560.5, 1561.7, 1563. , 1564.4, 1565.6, 1566.8, 1568., 1569.2, 1570.5, 1571.7, 1572.9])
		d_w = np.array([-1.00000000e-08, -8.00000000e-09, -6.00000000e-09, -4.00000000e-09, -2.00000000e-09, -5.29395592e-23,  2.00000000e-09,  4.00000000e-09, 6.00000000e-09,  8.00000000e-09,  1.00000000e-08])
		dlam_dp, p_0 = np.polyfit(periods, lam_p, 1)
		dlam_dw, w_0 = np.polyfit(d_w, lam, 1)
		wvl = dlam_dp*period + p_0 + dlam_dw*dw

		return wvl


	# This finds optimal period and widths combination for a targeted ref. wavelength
	def optimizeParams(self, target_wvl):
		
		error = 20e-9
		new_error = 10e-9

		# Parameters gotten from regression
		dlam_dp = 2.853181818181853
		p_0     = 6.423545454545346e-07
		dlam_dw = 0.6204545454543569

		# creating dummy device and estimating parameters through fit
		dummy = copy.copy(self)
		dummy.target_wvl = None # Most important line ever ;)
		dummy.N = 500 # Doesn't really change centre wavelength and saves time
		dummy.period = np.round((target_wvl - p_0)/dlam_dp/self.period_chirp_step)*self.period_chirp_step
		dw = np.round((target_wvl - dlam_dp*dummy.period - p_0)/dlam_dw, 9)
		dummy.w1 = dummy.w1 + dw 
		dummy.w2 = dummy.w2 + dw 

		dummy.wvl_range = [target_wvl-30e-9, target_wvl+30e-9]
		dummy.resolution = 50

		# Iterating until best combination is found
		run = True
		while run:
			error = new_error
			dummy.simulate(bar=False)
			dummy.getPerformance()
			ref = dummy.performance[0][1]*1e-9 # the ref wavelength
			new_error = ref - target_wvl

			if abs(new_error) > abs(error):
				if new_error > 0:
					dummy.w1 = dummy.w1[0] - self.w_chirp_step
					dummy.w2 = dummy.w2[0] - self.w_chirp_step
				elif new_error < 0:
					dummy.w1 = dummy.w1[0] + self.w_chirp_step
					dummy.w2 = dummy.w2[0] + self.w_chirp_step
				run = False

			elif abs(new_error) < abs(error):
				# print(new_error, dummy.w1, dummy.w2)
				if new_error > 0:
					dummy.w1 = dummy.w1[0] - self.w_chirp_step
					dummy.w2 = dummy.w2[0] - self.w_chirp_step
				elif new_error < 0:
					dummy.w1 = dummy.w1[0] + self.w_chirp_step
					dummy.w2 = dummy.w2[0] + self.w_chirp_step

			else:
				run = False

		if isinstance(dummy.w1, float):
			return dummy.period, dummy.w1, dummy.w2
		else:
			return dummy.period, dummy.w1[0], dummy.w2[0]


	# This finds the best chirp profile to smoothly scan reflection wavelengths
	def optimizeChirp(self, start_wvl, end_wvl, bar=True):
		
		ref_wvl = np.linspace(start_wvl, end_wvl, self.N_seg)
		self.period_profile = np.zeros(self.N_seg)
		self.w1_profile = np.zeros(self.N_seg)
		self.w2_profile = np.zeros(self.N_seg)

		if bar:
			progressbar_width = self.N_seg
			self.printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
			i=0

		for n in range(self.N_seg):
			self.period_profile[n], self.w1_profile[n], self.w2_profile[n] = self.optimizeParams(ref_wvl[n])

			if bar:
				i += 1
				clc()
				print("Optimizing chirp profile...")
				self.printProgressBar(i, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)


	def getChirpProfile(self, plot=False):

		if self.target_wvl is None: # if no chirp optimization is used
			# Period chirp
			period = self.period
			if isinstance(self.period, float):
				period = [self.period] # convert to list

			periods = np.arange(period[0],period[-1]+1e-9,self.period_chirp_step)
			num_per = round((period[-1]-period[0])/self.period_chirp_step + 1)
			l_seg = np.ceil(self.N_seg/num_per)
			period_profile = np.repeat(periods,l_seg)
			self.period_profile = period_profile
			if plot:
				plt.figure()
				plt.plot(self.period_profile*1e9,"o-")
				plt.xlabel("Apodization segment")
				plt.ylabel("Period (nm)")

			# Waveguide width chirp
			if isinstance(self.w1, float):
				self.w1 = [self.w1] # convert to list
			self.w1_profile = np.linspace(self.w1[0],self.w1[-1],self.N_seg)
			self.w1_profile = np.round(self.w1_profile,9)

			if isinstance(self.w2, float):
				self.w2 = [self.w2] # convert to list
			self.w2_profile = np.linspace(self.w2[0],self.w2[-1],self.N_seg)
			self.w2_profile = np.round(self.w2_profile,9)

		else: # if chirp optimization is used
			self.optimizeChirp(self.target_wvl[0], self.target_wvl[-1])

		if plot:
			plt.figure()
			plt.plot(self.period_profile*1e9,"o-")
			plt.xlabel("Apodization segment")
			plt.ylabel("Period (nm)")

			plt.figure()
			plt.plot(self.w1_profile,"o-")
			plt.plot(self.w2_profile,"o-")
			plt.title("Width Chirp Profile")
			plt.show()


	# a new chirping technique: constant sides for better squareness
	def chirpV2(self, frac=1/4):
		# period
		p1 = self.period[0]*np.ones(int(self.N_seg*frac))
		p2 = np.linspace(self.period[0], self.period[-1], int(self.N_seg*(1-2*frac)))
		p2 = np.round(p2/2,9)*2
		p3 = self.period[-1]*np.ones(int(self.N_seg*frac))

		p1=np.append(p1,p2)
		p1=np.append(p1,p3)
		while p1.size < self.N_seg:
			p1=np.append(p1,self.period[-1])

		self.period_profile = p1


		w1 = self.w1[0]*np.ones(int(self.N_seg*frac))
		w2 = np.linspace(self.w1[0], self.w1[-1], int(self.N_seg*(1-2*frac)))
		w2 = np.round(w2,9)
		w3 = self.w1[-1]*np.ones(int(self.N_seg*frac))

		w1=np.append(w1,w2)
		w1=np.append(w1,w3)
		while w1.size < self.N_seg:
			w1=np.append(w1,self.w1[-1])

		self.w1_profile = w1
	

		w1 = self.w2[0]*np.ones(int(self.N_seg*frac))
		w2 = np.linspace(self.w2[0], self.w2[-1], int(self.N_seg*(1-2*frac)))
		w2 = np.round(w2,9)
		w3 = self.w2[-1]*np.ones(int(self.N_seg*frac))

		w1=np.append(w1,w2)
		w1=np.append(w1,w3)
		while w1.size < self.N_seg:
			w1=np.append(w1,self.w2[-1])

		self.w2_profile = w1
		print(w1)

	# end section on chirp
	# --------------\


	def propagate(self, bar):
		# initiate arrays
		T = np.zeros((1, self.resolution),dtype=complex)
		R = np.zeros((1, self.resolution),dtype=complex)
		T_co = np.zeros((1, self.resolution),dtype=complex)
		R_co = np.zeros((1, self.resolution),dtype=complex)
		
		E_Thru = np.zeros((1, self.resolution),dtype=complex)
		E_Drop = np.zeros((1, self.resolution),dtype=complex)

		LeftRightTransferMatrix = np.zeros((4,4,self.resolution),dtype=complex)
		TopDownTransferMatrix = np.zeros((4,4,self.resolution),dtype=complex)
		InOutTransferMatrix = np.zeros((4,4,self.resolution),dtype=complex)

		kappa_apod = self.getApodProfile()

		mode_kappa_a1=1
		mode_kappa_a2=0 #no initial cross coupling
		mode_kappa_b2=1
		mode_kappa_b1=0

		j = cmath.sqrt(-1)      # imaginary

		alpha_e = 100*self.alpha/10*math.log(10)

		if bar:
			progressbar_width = self.resolution
			# Initial call to print 0% progress
			self.printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
		        
		# Propagation 
		# i: wavelength, related to self.resolution
		# j: profiles along grating, related to self.N_seg	
       
		for ii in range(self.resolution):
			if bar:
				clc()
				print("Propagating along grating...")
				self.printProgressBar(ii + 1, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

			l_0 = 0
			for n in range(self.N_seg):

				l_seg = self.N/self.N_seg * self.period_profile[n]			

				kappa_12 = self.apod_profile[n]
				kappa_21 = np.conj(kappa_12);
				kappa_11 = self._antiRefCoeff * self.apod_profile[n]
				kappa_22 = self._antiRefCoeff * self.apod_profile[n]

				beta_del_1 = self.beta1_profile[ii,n] - math.pi/self.period_profile[n]  - j*alpha_e/2
				beta_del_2 = self.beta2_profile[ii,n] - math.pi/self.period_profile[n]  - j*alpha_e/2

				S_1=[  [j*beta_del_1, 0, 0, 0], [0, j*beta_del_2, 0, 0],
				       [0, 0, -j*beta_del_1, 0],[0, 0, 0, -j*beta_del_2]]

				# S2 = transfert matrix
				S_2=  [[-j*beta_del_1,  0,  -j*kappa_11*np.exp(j*2*beta_del_1*l_0),  -j*kappa_12*np.exp(j*(beta_del_1+beta_del_2)*l_0)],
				       [0,  -j*beta_del_2,  -j*kappa_12*np.exp(j*(beta_del_1+beta_del_2)*l_0),  -j*kappa_22*np.exp(j*2*beta_del_2*l_0)],
				       [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1*l_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1+beta_del_2)*l_0),  j*beta_del_1,  0],
				       [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1+beta_del_2)*l_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2*l_0),  0,  j*beta_del_2]]

				P0=np.matmul(scipy.linalg.expm(np.asarray(S_1)*l_seg),scipy.linalg.expm(np.asarray(S_2)*l_seg))
				if n == 0:
				    P1 = P0*1
				else:
				    P1 = np.matmul(P0,P)
				P = P1

				l_0 = l_0 + l_seg

			    
			LeftRightTransferMatrix[:,:,ii] = P
			# Calculating In Out Matrix
			# Matrix Switch, flip inputs 1&2 with outputs 1&2
			H = self.switchTop(P)
			InOutTransferMatrix[:,:,ii] = H

			# Calculate Top Down Matrix
			P2 = P
			# switch the order of the inputs/outputs
			P2=np.vstack((P2[3][:], P2[1][:], P2[2][:], P2[0][:])) # switch rows
			P2=self.swap_cols(P2,1,2) # switch columns
			# Matrix Switch, flip inputs 1&2 with outputs 1&2
			P3 = self.switchTop( P2 )
			# switch the order of the inputs/outputs
			P3=np.vstack((P3[3][:], P3[0][:], P3[2][:], P3[1][:])) # switch rows
			P3=self.swap_cols(P3,2,3) # switch columns
			P3=self.swap_cols(P3,1,2) # switch columns

			TopDownTransferMatrix[:,:,ii] = P3
			T[0,ii] = H[0,0]*mode_kappa_a1+H[0,1]*mode_kappa_a2
			R[0,ii] = H[3,0]*mode_kappa_a1+H[3,1]*mode_kappa_a2

			T_co[0,ii] = H[1,0]*mode_kappa_a1+H[1,0]*mode_kappa_a2
			R_co[0,ii] = H[2,0]*mode_kappa_a1+H[2,1]*mode_kappa_a2

			E_Thru[0,ii] = mode_kappa_a1*T[0,ii]+mode_kappa_a2*T_co[0,ii]
			E_Drop[0,ii] = mode_kappa_b1*R_co[0,ii] + mode_kappa_b2*R[0,ii]

		# return results
		self.E_thru = E_Thru
		self.thru = 10*np.log10(np.abs(self.E_thru[0,:])**2)

		self.E_drop = E_Drop
		self.drop = 10*np.log10(np.abs(self.E_drop[0,:])**2)

		self.TransferMatrix = LeftRightTransferMatrix

	def flipProfiles(self): # flips the cdc
			self.beta1_profile = np.flip(self.beta1_profile)
			self.beta2_profile = np.flip(self.beta2_profile)
			self.period_profile = np.flip(self.period_profile)

	def cascade(self):
		if self.stages > 1:
			thru1, drop1 = self.thru, self.drop
			self.flipProfiles()
			self.propagate(True)
			thru2, drop2 = self.thru, self.drop
			for _ in range(self.stages):
				if _%2 == 0:
					drop, thru = drop2, thru2
				else:
					drop, thru = drop1, thru1

				self.thru = self.thru + thru
				self.drop = self.drop + drop
			self.flipProfiles() # Return to original one

	def simulate(self, bar=True):
		self.getApodProfile()
		self.getChirpProfile()
		self.getPropConstants(bar)
		self.propagate(bar)
		self.cascade()

	def getPerformance(self):
		if self.E_thru is not None:

			# bandwidth and centre wavelength
			dropMax = max(self.drop)
			drop3dB = self.wavelength[self.drop > dropMax - 3]
			ref_wvl = (drop3dB[-1] + drop3dB[0]) /2
			# TODO: something to discard sidelobes from 3-dB bandwidth
			bandwidth = drop3dB[-1] - drop3dB[0]

			# Top flatness assessment
			dropBand = self.drop[self.drop > dropMax - 3]
			avg = np.mean(dropBand)
			std = np.std(dropBand)

			# Extinction ratio
			ER = -1

			# Smoothness
			smoothness = -1

			self.performance = \
				[("Ref. Wvl" , np.round(ref_wvl*1e9,1)           ,  "nm"), \
				("BW"              , np.round(bandwidth*1e9,1)  ,  "nm"), \
				("Max Ref."         , np.round(dropMax,2)          ,  "dB"), \
				("Avg Ref."     , np.round(avg,2)              ,  "dB"), \
				("Std Dev."     , np.round(std,2)              ,  "dB")] \
				# ("Exctinction Ratio"      , np.round(ER,1)               ,  "dB"), \
				# ("Smoothness"             , np.round(smoothness)       ,  " " )]


	# Display Plots and figures of merit 
	def displayResults(self, advanced=False):

		# clc()
		# print("Displaying results.")

		self.getPerformance()


		fig = plt.figure(figsize=(9,6))
		grid = plt.GridSpec(6,3)

		plt.subplot(grid[0:2,0])
		plt.title("Grating Profiles")
		plt.plot(self.apod_profile/1000,".-")
		plt.xticks([])
		plt.ylabel("Coupling (/mm)")
		plt.tick_params(axis='y', direction="in", right=True)
		plt.text(self.N_seg/2,self.kappa/4/1000,"a = "+str(self.a),ha="center")

		plt.subplot(grid[2:4,0])
		plt.plot(self.period_profile*1e9,".-")
		plt.xticks([])
		plt.ylabel("Pitch (nm)")
		plt.tick_params(axis='y', direction="in", right=True)

		plt.subplot(grid[4,0])
		plt.plot(self.N/self.N_seg*np.arange(0,self.w1_profile.size),self.w1_profile*1e9,".-",label="wg 1")
		plt.ylabel("w1 (nm)")
		plt.tick_params(axis='y', direction="in", right=True)
		plt.xticks([])

		plt.subplot(grid[5,0])
		plt.plot(self.N/self.N_seg*np.arange(0,self.w2_profile.size),self.w2_profile*1e9,".-",label="wg 2")
		plt.xlabel("Period Along Grating")
		plt.ylabel("w2 (nm)")
		plt.tick_params(axis='y', direction="in", right=True)	


		plt.subplot(grid[0:2,1])
		plt.title("Specifications")
		numElems = 6
		plt.axis([0,1,-numElems+1,1])
		plt.text(0.5,-0,"N : " + str(self.N),fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-1,"N_seg : " + str(self.N_seg),fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-2,"a : " + str(self.a),fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-3,"p: " + str(self.period)+" m",fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-4,"w1 : " + str(self.w1)+" m",fontsize=11,ha="center",va="bottom")
		plt.text(0.5,-5,"w2 : " + str(self.w2)+" m",fontsize=11,ha="center",va="bottom")
		plt.xticks([])
		plt.yticks([])
		plt.box(False)


		plt.subplot(grid[0:2,2])
		plt.title("Performance")
		numElems = np.size(self.performance)/3
		plt.axis([0,1,-numElems+1,1])
		for i in np.arange(0,5):
			plt.text(0.5,-i,self.performance[i][0]+" : ",fontsize=11,ha="right",va="bottom")
			plt.text(0.5,-i,str(self.performance[i][1])+" "+self.performance[i][2],fontsize=11,ha="left",va="bottom")
		plt.xticks([])
		plt.yticks([])
		plt.box(False)

		
		plt.subplot(grid[2:,1:])
		plt.plot(self.wavelength*1e9, self.thru, label="Thru port")
		plt.plot(self.wavelength*1e9, self.drop, label="Drop port")
		plt.legend()
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Response (dB)")
		plt.tick_params(axis='y', which='both', labelleft=False, labelright=True, \
						direction="in", right=True)
		plt.tick_params(axis='x', top=True)


		plt.show()


