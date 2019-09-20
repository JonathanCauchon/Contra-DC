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

class Waveguide():
	def __init__(self, w=500e-9, h=220e-9, neff=2.5, Dneff=-969000, simulate=False):

		# General case
		self.w = w
		self.h = h
		self.neff = neff
		self.Dneff = Dneff

		# special cases used for standard contraDC
		if (self.w == 560e-9 and self.h == 220e-9):
			self.neff = 2.5316
			self.Dneff = -969700

		elif (self.w == 440e-9 and self.h == 220e-9):
			self.neff = 2.3404
			self.Dneff = -1220800

		if simulate == True:
			self.simulate()

	def simulate(self):
		# This would use Mode Solutions to extract neff, Dneff
		pass



class ChirpedContraDC():
	def __init__(self, N = 1000, period = 322e-9, DC = 0.5, a = 12, kappa = 48000, T = 300, \
		resolution = 300, N_seg = 50, wvl_range = [1530e-9,1580e-9], central_wvl = 1550e-9, \
		alpha = 10, stages = 1, w1 = .56e-6, w2 = .44e-6):

		# Instantiatable parameters
		self.N = N
		self.period = period
		self.DC = DC	# Duty cycle
		self.a = a
		self.kappa = kappa
		self.T = T
		self.resolution = resolution
		self.N_seg = N_seg
		self.central_wvl = central_wvl
		self.alpha = alpha # dB/cm
		self.stages = stages # number of cascaded devices
		self.wavelength = np.linspace(wvl_range[0], wvl_range[1], resolution)

		self.w1, self.w2 = w1, w2

		self.antiRefCoeff = 0.01

		# Constants
		# c = 299792458           #[m/s]

		# Chrip
		self.period_chirp_step = 2e-9 # To comply with GDS resolution
		# self.period_profile = self.getChirpProfile()
		self.w_chirp_step = 5e-9
		# Jflag change the above for 1 nm, eventually
		# self.DC_profile = self.


		# Protecting the model against user-induced inconsistancies
		# Gives warnings, errors, makes corrections

		# if N is a multiple of N_seg
		if self.N%self.N_seg:
			print("Number of periods (N) should be an integer multiple of the number of segments (N_seg).")
			self.N_seg = 50
			print("Number of segments was changed to "+str(self.N_seg)+".")



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
		"""
		Call in a loop to create terminal progress bar
			@params:
			iteration   - Required  : current iteration (Int)
			total       - Required  : total iterations (Int)
			prefix      - Optional  : prefix string (Str)
			suffix      - Optional  : suffix string (Str)
			decimals    - Optional  : positive number of decimals in percent complete (Int)
			length      - Optional  : character length of bar (Int)
			fill        - Optional  : bar fill character (Str)
			"""
		percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
		filledLength = int(length * iteration // total)
		bar = fill * filledLength + '-' * (length - filledLength)
		print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
		# Print New Line on Complete
		if iteration == total: 
		    print()

	# Calculate beta at phase match
	# def getBetaWav(self, beta): 
	# 	f = 2*math.pi/beta 							# = grating period at phase match
	# 	minimum = min(abs(f-np.mean(self.period))) 				#index of closest value
	# 	idx = np.where(abs(f-np.mean(self.period)) == minimum)
	# 	betaWav = self.wavelength.item(idx[0][0])
	# 	return betaWav

	# get effective indices vs wg width based on MODE simulations
	# def getEffectiveIndex(self,w1,w2,plot=False):
	def getPropConstants(self,w1,w2,plot=False):

		#-------------/ OLD, better version below
		# file = "indices.txt"
		# m = np.loadtxt(file)

		# # checking which width combination fits
		# w1 = np.round(w1,10)
		# w2 = np.round(w2,10)
		# idx_w1 = np.where(m==w1,1,0)[:,0]
		# idx_w2 = np.where(m==w2,1,0)[:,1]
		# idx = np.where(idx_w1*idx_w2==1)
		# m_ = m[idx] # array containing only good widths

		# p1, p2 = np.polyfit(m_[:,2], m_[:,3],2), np.polyfit(m_[:,2], m_[:,4],2)
		# n1 = p1[0]*self.wavelength**2 + p1[1]*self.wavelength + p1[2]
		# n2 = p2[0]*self.wavelength**2 + p2[1]*self.wavelength + p2[2]

		# if plot:
		# 	plt.figure()
		# 	plt.plot(m_[:,2]*1e9, m_[:,3],"o-",label="Simulated")
		# 	plt.plot(wavelength*1e9,n1,".",label="Fitted")
		# 	plt.legend()
		# 	plt.show()
		#-----------\ OLD

		# This performs a 3d interpolation to estimate effective indices
		n1 = np.reshape(np.loadtxt("Database/neff/neff_1.txt"),(5,5,5))
		n2 = np.reshape(np.loadtxt("Database/neff/neff_2.txt"),(5,5,5))
		w1_w2_wvl = np.loadtxt("Database/neff/w1_w2_lambda.txt")
		n1_ = np.zeros((self.wavelength.size,1))
		n2_ = np.zeros((self.wavelength.size,1))
		for i in range(self.wavelength.size):
			n1_[i] = scipy.interpolate.interpn(w1_w2_wvl, n1, [w1,w2,self.wavelength[i]])
			n2_[i] = scipy.interpolate.interpn(w1_w2_wvl, n2, [w1,w2,self.wavelength[i]])

		neffwg1, neffwg2 = n1_, n2_ # the interpolated values

	# def getPropConstants(self):
		# neffwg1 = self.wg1.neff
		# Dneffwg1 = self.wg1.Dneff
		# neffwg2 = self.wg2.neff
		# Dneffwg2 = self.wg2.Dneff

		T0 = 300
		dneffdT = 1.87E-04      #[/K] assuming dneff/dn=1 (well confined mode)
		neffThermal = dneffdT*(self.T-T0)

		# neff_a_data = neffwg1+Dneffwg1*(self.wavelength-self.central_wvl) + neffThermal
		# neff_b_data = neffwg2+Dneffwg2*(self.wavelength-self.central_wvl) + neffThermal
		neff_a_data = neffwg1 + neffThermal
		neff_b_data = neffwg2 + neffThermal

		Lambda_data_left = self.wavelength
		Lambda_data_right = self.wavelength

		# beta_data_left=2*math.pi/Lambda_data_left*neff_a_data
		# beta_data_right=2*math.pi/Lambda_data_right*neff_b_data
		beta_left=2*math.pi/Lambda_data_left*neff_a_data
		beta_right=2*math.pi/Lambda_data_right*neff_b_data

		# beta_left = np.interp(self.wavelength, Lambda_data_left, beta_data_left)
		# beta_right = np.interp(self.wavelength, Lambda_data_right, beta_data_right)    

		# beta12Wav = self.getBetaWav(beta_left+beta_right)
		# beta1Wav = self.getBetaWav(beta_left)
		# beta2Wav = self.getBetaWav(beta_right)

		return beta_left, beta_right


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
		n=np.arange(self.N_seg)

		return kappa_apod


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
		if plot == True:
			plt.figure()
			plt.plot(self.period_profile*1e9,"o-")
			plt.xlabel("Apodization segment")
			plt.ylabel("Period (nm)")
			plt.show()

		# Waveguide width chirp
		if isinstance(self.w1, float):
			self.w2 = [self.w1] # convert to list
		numW1 = round((self.w1[-1]-self.w1[0])/self.w_chirp_step + 1)
		numW1 = abs(numW1)
		l_seg = np.ceil(self.N_seg/numW1)
		w1s = np.linspace(self.w1[0],self.w1[-1],numW1)
		self.w1_profile = np.repeat(w1s,l_seg)

		if isinstance(self.w2, float):
			self.w2 = [self.w2] # convert to list
		numW2 = round((self.w2[-1]-self.w2[0])/self.w_chirp_step + 1)
		numW2 = abs(numW2)
		l_seg = np.ceil(self.N_seg/numW2)
		w2s = np.linspace(self.w2[0],self.w2[-1],numW2)
		self.w2_profile = np.repeat(w2s,l_seg)


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
		        
		for ii in range(self.resolution):
			#Update Bar
			self.printProgressBar(ii + 1, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

			for n in range(self.N_seg):
				beta_left, beta_right = self.getPropConstants(self.w1_profile[n], self.w2_profile[n])
				l_seg = self.N*np.mean(self.period)/self.N_seg				
				l_0 = n*l_seg

				kappa_12 = kappa_apod.item(n)
				kappa_21 = np.conj(kappa_12);
				kappa_11 = self.antiRefCoeff*kappa_apod.item(n)
				kappa_22 = self.antiRefCoeff*kappa_apod.item(n)


				#-------------------------------------------------#
				beta_del_1=beta_left - math.pi/self.period_profile[n]  - j*alpha_e/2
				beta_del_2=beta_right - math.pi/self.period_profile[n]  - j*alpha_e/2
				#-------------------------------------------------#	

				# S1 = Matrix of propagation in each guide & direction
				S_1=[  [j*beta_del_1.item(ii), 0, 0, 0], [0, j*beta_del_2.item(ii), 0, 0],
				       [0, 0, -j*beta_del_1.item(ii), 0],[0, 0, 0, -j*beta_del_2.item(ii)]]

				# S2 = transfert matrix
				S_2=  [[-j*beta_del_1.item(ii),  0,  -j*kappa_11*np.exp(j*2*beta_del_1.item(ii)*l_0),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0)],
				       [0,  -j*beta_del_2.item(ii),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0),  -j*kappa_22*np.exp(j*2*beta_del_2.item(ii)*l_0)],
				       [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1.item(ii)*l_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0),  j*beta_del_1.item(ii),  0],
				       [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2.item(ii)*l_0),  0,  j*beta_del_2.item(ii)]]

				P0=np.matmul(scipy.linalg.expm(np.asarray(S_1)*l_seg),scipy.linalg.expm(np.asarray(S_2)*l_seg))
				if n == 0:
				    P1 = P0*1
				else:
				    P1 = np.matmul(P0,P)
				P = P1
			    
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

		#%% return results
		self.E_Thru = E_Thru
		self.E_Drop = E_Drop
		self.TransferMatrix = LeftRightTransferMatrix

	def simulate(self):
		self.getChirpProfile()
		self.propagate()

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
				[("Reflection Wavelength" , int(ref_wvl*1e9)           ,  "nm"), \
				("Bandwidth"              , np.round(bandwidth*1e9,1)  ,  "nm"), \
				("Max Reflection"         , np.round(dropMax,2)          ,  "dB"), \
				("Average Reflection"     , np.round(avg,2)              ,  "dB"), \
				("Standard Deviation"     , np.round(std,2)              ,  "dB"), \
				("Exctinction Ratio"      , np.round(ER,1)               ,  "dB"), \
				("Smoothness"             , np.round(smoothness)       ,  " " )]



	# Display Plots and figures of merit 
	def displayResults(self):
		self.getPerformance()
		thruAmplitude = 10*np.log10(np.abs(self.E_Thru[0,:])**2)
		dropAmplitude = 10*np.log10(np.abs(self.E_Drop[0,:])**2)


		x = np.linspace(0, 2*np.pi, 400)
		y = np.sin(x**2)


		fig = plt.figure(figsize=(9,6))
		grid = plt.GridSpec(3,3)

		plt.subplot(grid[0,0])
		plt.title("Grating Profiles")
		plt.plot(np.arange(0,self.N_seg),self.getApodProfile()/1000)
		plt.xticks([])
		plt.ylabel("Coupling (/mm)")
		plt.tick_params(axis='y', direction="in", right=True)
		plt.text(25,self.kappa/4/1000,"a = "+str(self.a),ha="center")

		plt.subplot(grid[1,0])
		plt.plot(self.period_profile*1e9)
		plt.xticks([])
		plt.ylabel("Pitch (nm)")
		plt.tick_params(axis='y', direction="in", right=True)

		plt.subplot(grid[2,0])
		plt.plot(self.N/self.N_seg*np.arange(0,self.w1_profile.size),self.w1_profile*1e9,label="wg 1")
		plt.plot(self.N/self.N_seg*np.arange(0,self.w2_profile.size),self.w2_profile*1e9,label="wg 2")
		plt.xlabel("Period Along Grating")
		plt.ylabel("WG Width (nm)")
		plt.legend()
		plt.tick_params(axis='y', direction="in", right=True)

		plt.subplot(grid[0,1:])
		plt.title("Filter Performance")
		numElems = np.size(self.performance)/3
		plt.axis([0,1,-numElems+1,1])
		for i in np.arange(0,7):
			plt.text(0.6,-i,self.performance[i][0]+" : ",fontsize=11,ha="right",va="bottom")
			plt.text(0.6,-i,str(self.performance[i][1])+" "+self.performance[i][2],fontsize=11,ha="left",va="bottom")
		plt.xticks([])
		plt.yticks([])

		plt.subplot(grid[1:,1:])
		plt.plot(self.wavelength*1e9,thruAmplitude,label="Thru port")
		plt.plot(self.wavelength*1e9,dropAmplitude,label="Drop port")
		plt.legend()
		plt.xlabel("Wavelength (nm)")
		plt.ylabel("Response (dB)")
		plt.tick_params(axis='y', which='both', labelleft=False, labelright=True, \
						direction="in", right=True)

		plt.show()

