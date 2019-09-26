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


"""

"""   Notes
- Experimental vs simulated:
	Center wavelength: experimental is 8 nm higher than simulated center wvl

"""

from Modules import *

def clc():
	print ("\n"*10)

# class Waveguide():
# 	def __init__(self, w=500e-9, h=220e-9, neff=2.5, Dneff=-969000, simulate=False):

# 		# General case
# 		self.w = w
# 		self.h = h
# 		self.neff = neff
# 		self.Dneff = Dneff

# 		# special cases used for standard contraDC
# 		if (self.w == 560e-9 and self.h == 220e-9):
# 			self.neff = 2.5316
# 			self.Dneff = -969700

# 		elif (self.w == 440e-9 and self.h == 220e-9):
# 			self.neff = 2.3404
# 			self.Dneff = -1220800

# 		if simulate == True:
# 			self.simulate()

# 	def simulate(self):
# 		# This would use Mode Solutions to extract neff, Dneff
# 		pass



class ChirpedContraDC():
	def __init__(self, N = 1000, period = 322e-9, DC = 0.5, a = 12, kappa = 48000, T = 300, \
		resolution = 300, N_seg = 50, wvl_range = [1530e-9,1580e-9], central_wvl = 1550e-9, \
		alpha = 10, stages = 1, w1 = .56e-6, w2 = .44e-6):

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
		# Note that gap is set to 100 nm


		# Constants
		self._antiRefCoeff = 0.01
		self.period_chirp_step = 2e-9 # To comply with GDS resolution
		self.w_chirp_step = 1e-9


		# Protecting the model against user-induced inconsistancies
		# Gives warnings, errors, makes corrections

		# Check if N is a multiple of N_seg
		if self.N%self.N_seg:
			print("Number of periods (N) should be an integer multiple of the number of segments (N_seg).")
			self.N_seg = 50
			print("Number of segments was changed to "+str(self.N_seg)+".")


	# Property functions: changing one property automatically affects others
	@ property
	def wavelength(self):
		return np.linspace(self.wvl_range[0], self.wvl_range[1], self.resolution)

	@ wavelength.setter
	def wavelength(self):
		self.wavelength = np.linspace(self.wvl_range[0], self.wvl_range[1], self.resolution)


	@ property
	def c(self):
		return 299792458

	@ c.setter
	def c(self, value):
		print("The value of c can't be changed")


	#%% linear algebra numpy manipulation functions
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
	def getPropConstants(self,plot=False):
		
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

		progressbar_width = self.resolution
		clc()
		print("Calculating propagation constants...")		
		self.printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

		for i in range(self.resolution): # i=lambda, j=z

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


	def getChirpProfile(self, plot=False):

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

		if plot:
			plt.figure()
			plt.plot(self.w1_profile,"o-")
			plt.plot(self.w2_profile,"o-")
			plt.title("Width Chirp Profile")
			plt.show()


	def propagate(self):
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

		progressbar_width = self.resolution
		# Initial call to print 0% progress
		self.printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
		        
		# Propagation 
		# i: wavelength, related to self.resolution
		# j: profiles along grating, related to self.N_seg	
       
		for ii in range(self.resolution):
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
		self.E_Thru = E_Thru
		self.E_Drop = E_Drop
		self.TransferMatrix = LeftRightTransferMatrix

	def flipProfiles(self): # flips the cdc
			self.beta1_profile = np.flip(self.beta1_profile)
			self.beta2_profile = np.flip(self.beta2_profile)
			self.period_profile = np.flip(self.period_profile)

	def cascade(self):
		if self.stages > 1:
			thru1, drop1 = self.E_Thru, self.E_Drop
			self.flipProfiles()
			self.propagate()
			thru2, drop2 = self.E_Thru, self.E_Drop
			for _ in range(self.stages):
				if _%2 == 0:
					drop, thru = drop2, thru2
				else:
					drop, thru = drop1, thru1

				self.E_Thru = self.E_Thru*thru
				self.E_Drop = self.E_Drop*drop
			self.flipProfiles() # To keep the original one

	def simulate(self):
		self.getApodProfile()
		self.getChirpProfile()
		self.getPropConstants()
		self.propagate()
		self.cascade()

	def getPerformance(self):
		if self.E_Thru is not None:
			self.thru = 10*np.log10(np.abs(self.E_Thru[0,:])**2)
			self.drop = 10*np.log10(np.abs(self.E_Drop[0,:])**2)

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
				[("Reflection Wavelength" , np.round(ref_wvl*1e9,1)           ,  "nm"), \
				("Bandwidth"              , np.round(bandwidth*1e9,1)  ,  "nm"), \
				("Max Reflection"         , np.round(dropMax,2)          ,  "dB"), \
				("Average Reflection"     , np.round(avg,2)              ,  "dB"), \
				("Standard Deviation"     , np.round(std,2)              ,  "dB"), \
				("Exctinction Ratio"      , np.round(ER,1)               ,  "dB"), \
				("Smoothness"             , np.round(smoothness)       ,  " " )]


	# Display Plots and figures of merit 
	def displayResults(self):

		clc()
		print("Displaying results.")

		self.getPerformance()
		thruAmplitude = 10*np.log10(np.abs(self.E_Thru[0,:])**2)
		dropAmplitude = 10*np.log10(np.abs(self.E_Drop[0,:])**2)

		x = np.linspace(0, 2*np.pi, 400)
		y = np.sin(x**2)

		fig = plt.figure(figsize=(9,6))
		grid = plt.GridSpec(6,3)

		plt.subplot(grid[0:2,0])
		plt.title("Grating Profiles")
		plt.plot(np.arange(0,self.N_seg),self.apod_profile/1000)
		plt.xticks([])
		plt.ylabel("Coupling (/mm)")
		plt.tick_params(axis='y', direction="in", right=True)
		plt.text(self.N_seg/2,self.kappa/4/1000,"a = "+str(self.a),ha="center")

		plt.subplot(grid[2:4,0])
		plt.plot(self.period_profile*1e9)
		plt.xticks([])
		plt.ylabel("Pitch (nm)")
		plt.tick_params(axis='y', direction="in", right=True)

		plt.subplot(grid[4,0])
		plt.plot(self.N/self.N_seg*np.arange(0,self.w1_profile.size),self.w1_profile*1e9,label="wg 1")
		plt.ylabel("WG Widths (nm)")
		plt.xticks([])

		plt.subplot(grid[5,0])
		plt.plot(self.N/self.N_seg*np.arange(0,self.w2_profile.size),self.w2_profile*1e9,label="wg 2")
		plt.xlabel("Period Along Grating")
		plt.tick_params(axis='y', direction="in", right=True)	


		plt.subplot(grid[0:2,1:])
		plt.title("Filter Performance")
		numElems = np.size(self.performance)/3
		plt.axis([0,1,-numElems+1,1])
		for i in np.arange(0,7):
			plt.text(0.6,-i,self.performance[i][0]+" : ",fontsize=11,ha="right",va="bottom")
			plt.text(0.6,-i,str(self.performance[i][1])+" "+self.performance[i][2],fontsize=11,ha="left",va="bottom")
		plt.xticks([])
		plt.yticks([])

		plt.subplot(grid[2:,1:])
		plt.plot(self.wavelength*1e9,thruAmplitude,label="Thru port")
		plt.plot(self.wavelength*1e9,dropAmplitude,label="Drop port")
		plt.legend()
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Response (dB)")
		plt.tick_params(axis='y', which='both', labelleft=False, labelright=True, \
						direction="in", right=True)
		plt.tick_params(axis='x', top=True)

		plt.show()
		clc()
