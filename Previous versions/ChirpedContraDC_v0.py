"""
	ChripedContraDC.py
	Chirped contra-directional coupler model
	Chirp your CDC, engineer your response
	(Or let a computer engineer it for you)
	
	Based on Matlab model by Jonathan St-Yves
	as well as Python model by Mustafa Hammood

	Jonathan Cauchon, September 2019
"""

import cmath, math
import sys, os, time
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt

class ChripedContraDC():
	def __init__(self, N_corrugations=1000, temperature=300, start_wavelength=1.5e-6, end_wavelength=1.6e-6, \
		kappa=48000, resolution = 300, period=322e-9, a=12, N_seg=50, DC=0.5):

		# Properties set during instantiation:
		self.N_corrugations = N_corrugations
		self.temperature = temperature
		self.start_wavelength = start_wavelength
		self.end_wavelength = end_wavelength
		self.kappa = kappa
		self.resolution = resolution
		self.period = period
		self.a = a
		self.N_seg = N_seg
		self.DC = DC	# Duty cycle

		self.wavelength = np.linspace(self.start_wavelength, self.end_wavelength, self.resolution)

		# Waveguide properties 
		self.neffwg1 = 2.5316
		self.Dneffwg1 = -969700
		self.neffwg2 = 2.3404
		self.Dneffwg2 = -1220800

		self.centralWL_neff = 1550e-9
		self.alpha = 1000


		self.ApoFunc = np.exp(-np.linspace(0,1,1000)**2)
		self.mirror = 0
		self.antiRefCoeff = 0.01

		self.simulate()


	# Useful array manipulation
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

	def getBetaWav(self, beta):
		f = 2*math.pi / beta	# Phase match condition
		minimum = min(abs(f-np.mean(self.period)))
		idx = np.where(abs(f-np.mean(self.period)) == minimum)
		betaWav = self.wavelength.item(idx[0][0])
		return betaWav

	def getApodProfile(self):
		pass


	def propagate(self):
		# This will be just the transfer matrix part
		pass



	# Simulate contra-DC response
	def simulate(self): # 1 == left, 2==right



		"""

			def getPropConstants(self):

		"""

		# Constants
		c = 299792458
		dneffdT = 1.87e-4 # for silicon
		j = cmath.sqrt(-1) # imaginary
		room_temperature = 300
		
		# calculate waveguides propagation constants
		alpha_e = 100*self.alpha/10*math.log(10)
		neffThermal = dneffdT*(self.temperature-room_temperature)
    
		neff_a_data = self.neffwg1+self.Dneffwg1*(self.wavelength-self.centralWL_neff) + neffThermal
		neff_b_data = self.neffwg2+self.Dneffwg2*(self.wavelength-self.centralWL_neff) + neffThermal
		Lambda_data_left = Lambda_data_right = self.wavelength

		beta_data_left=2*math.pi/Lambda_data_left*neff_a_data
		beta_data_right=2*math.pi/Lambda_data_right*neff_b_data

		beta_left=np.interp(self.wavelength, Lambda_data_left, beta_data_left);
		beta_right=np.interp(self.wavelength, Lambda_data_right, beta_data_right);    
  
		beta12Wav = self.getBetaWav(beta_left+beta_right)
		beta1Wav = self.getBetaWav(beta_left)
		beta2Wav = self.getBetaWav(beta_right)

		T = R = T_co = R_co = E_Thru = E_Drop = np.zeros((1, self.resolution), dtype=complex)

		mode_kappa_a1 = mode_kappa_b2 = 1
		mode_kappa_a2 = mode_kappa_b1 = 0	# no initial cross coupling




		"""

			def getApodProfile(self):

		"""

		# Apodization & segmenting
		# l_seg = self.N_corrugations*np.mean(self.period)/self.N_seg
		n_apodization = np.arange(self.N_seg)+0.5
		# zaxis = (np.arange(self.N_seg))*l_seg

		if  self.a != 0:
			kappa_apodG = np.exp(-self.a*((n_apodization)-0.5*self.N_seg)**2/self.N_seg**2)
			ApoFunc = kappa_apodG

		profile = (ApoFunc-min(ApoFunc))/(max(ApoFunc)-(min(ApoFunc))) # normalizes the profile

		n_profile = np.linspace(0, self.N_seg, profile.size)
		profile = np.interp(n_apodization, n_profile, profile)

		# if False: #plot == True:
		# 	plt.figure(1)
		# 	plt.plot(zaxis*1e6, profile)
		# 	plt.ylabel('Apodization Profile (normalized)')
		# 	plt.xlabel('Length (um)')
		# 	plt.show()

		kappaMin = 0
		kappaMax = self.kappa

		kappa_apod=kappaMin+(kappaMax-kappaMin)*profile
		kappa_12max= max(kappa_apod)

		n = np.arange(self.N_seg)





		"""

			def propagate(self):

		"""

		# Beautiful Progress Bar
		# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
		# Thank Greenstick.

		progressbar_width = self.resolution
		# Initial call to print 0% progress
		# printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
		for ii in range(self.resolution):
			# for lambda, this should disapear for efficiency!!!

			# Update Bar
			# printProgressBar(ii + 1, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
		
			LeftRightTransferMatrix = TopDownTransferMatrix = InOutTransferMatrix \
			 = np.zeros((4, 4, self.resolution), dtype=complex)

			for n in range(self.N_seg):
				l_seg = self.N_corrugations*np.mean(self.period)/self.N_seg
				l_0 = (n)*l_seg

				kappa_12 = kappa_apod.item(n)
				kappa_21 = np.conj(kappa_12)
				kappa_11 = kappa_22 = self.antiRefCoeff*kappa_apod.item(n)



				#-------------------------------------------------#
				beta_del_1=beta_left - math.pi/self.period - j*alpha_e/2
				beta_del_2=beta_right - math.pi/self.period - j*alpha_e/2
				#-------------------------------------------------#



				# S1 = Matrix of propagation in each guide & direction
				S1 = [ [j*beta_del_1.item(ii), 0, 0, 0], [0, j*beta_del_2.item(ii), 0, 0],
				       [0, 0, -j*beta_del_1.item(ii), 0],[0, 0, 0, -j*beta_del_2.item(ii)] ]

				# S2 = transfert matrix
				S2 = [ [-j*beta_del_1.item(ii),  0,  -j*kappa_11*np.exp(j*2*beta_del_1.item(ii)*l_0),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0)],
				       [0,  -j*beta_del_2.item(ii),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0),  -j*kappa_22*np.exp(j*2*beta_del_2.item(ii)*l_0)],
				       [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1.item(ii)*l_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0),  j*beta_del_1.item(ii),  0],
				       [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*l_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2.item(ii)*l_0),  0,  j*beta_del_2.item(ii)] ]

				P0=np.matmul(scipy.linalg.expm(np.asarray(S1)*l_seg),scipy.linalg.expm(np.asarray(S2)*l_seg))
				if n == 0:
				    P1 = P0
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















cdc = ChripedContraDC(N_corrugations=500, period=320e-9, resolution=500, start_wavelength=1550e-9, end_wavelength=1650e-9)
cdc.simulate()

plt.figure(1)
plt.plot(np.squeeze(cdc.wavelength*1e9), np.squeeze(10*np.log10(np.absolute(cdc.E_Drop)**2)))
plt.show()
# print(cdc.thru)
# plt.plot(cdc.ApoFunc)
# plt.show()

# print(cdc.N_corruga)


# #%% import dependencies
# import cmath, math
# import sys, os, time
# import numpy as np
# import scipy.linalg
# import matplotlib.pyplot as plt

# #%% linear algebra numpy manipulation functions
# # Takes a 4*4 matrix and switch the first 2 inputs with first 2 outputs
# def switchTop( P ):
#     P_FF = np.asarray([[P[0][0],P[0][1]],[P[1][0],P[1][1]]])
#     P_FG = np.asarray([[P[0][2],P[0][3]],[P[1][2],P[1][3]]])
#     P_GF = np.asarray([[P[2][0],P[2][1]],[P[3][0],P[3][1]]])
#     P_GG = np.asarray([[P[2][2],P[2][3]],[P[3][2],P[3][3]]])
    
#     H1 = P_FF-np.matmul(np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1)),P_GF)
#     H2 = np.matmul(P_FG,np.linalg.matrix_power(P_GG,-1))
#     H3 = np.matmul(-np.linalg.matrix_power(P_GG,-1),P_GF)
#     H4 = np.linalg.matrix_power(P_GG,-1)
#     H = np.vstack((np.hstack((H1,H2)),np.hstack((H3,H4))))
    
#     return H

# # Swap columns of a given array
# def swap_cols(arr, frm, to):
#     arr[:,[frm, to]] = arr[:,[to, frm]]
#     return arr

# # Swap rows of a given array
# def swap_rows(arr, frm, to):
#     arr[[frm, to],:] = arr[[to, frm],:]
#     return arr
    
# #%% the bread and butter
# def contraDC_model(contraDC, simulation_setup, waveguides,plot = True):
    
#     #%% System constants Constants
#     c = 299792458           #[m/s]
#     dneffdT = 1.87E-04      #[/K] assuming dneff/dn=1 (well confined mode)
#     j = cmath.sqrt(-1)      # imaginary
    
#     neffwg1 = waveguides[0]
#     Dneffwg1 = waveguides[1]
#     neffwg2 = waveguides[2]
#     Dneffwg2 = waveguides[3]
    
#     ApoFunc=np.exp(-np.linspace(0,1,num=1000)**2)     #Function used for apodization (window function)

#     mirror = False                #makes the apodization function symetrical
#     N_seg = 100                   #Number of flat steps in the coupling profile
    
#     rch=0                        #random chirping, maximal fraction of index randomly changing each segment
#     lch=0                         #linear chirp across the length of the device
#     kch=0                        #coupling dependant chirp, normalized to the max coupling
    
#     neff_detuning_factor = 1
    
#     #%% calculate waveguides propagation constants
#     alpha_e = 100*contraDC.alpha/10*math.log(10)
#     neffThermal = dneffdT*(simulation_setup.deviceTemp-simulation_setup.chipTemp)

#     # Waveguides models
#     Lambda = np.linspace(simulation_setup.lambda_start, simulation_setup.lambda_end, num=simulation_setup.resolution)

#     neff_a_data = neffwg1+Dneffwg1*(Lambda-simulation_setup.central_lambda)
#     neff_a_data = neff_a_data*neff_detuning_factor+neffThermal
#     neff_b_data=neffwg2+Dneffwg2*(Lambda-simulation_setup.central_lambda)
#     neff_b_data = neff_b_data*neff_detuning_factor+neffThermal
#     Lambda_data_left=Lambda
#     Lambda_data_right=Lambda

#     beta_data_left=2*math.pi/Lambda_data_left*neff_a_data
#     beta_data_right=2*math.pi/Lambda_data_right*neff_b_data

#     #%% makes sense until HERE

#     beta_left=np.interp(Lambda, Lambda_data_left, beta_data_left); betaL=beta_left
#     beta_right=np.interp(Lambda, Lambda_data_right, beta_data_right); betaR=beta_right    
  
#     # Calculating reflection wavelenghts
#     period = contraDC.period
    
#     f= 2*math.pi/(beta_left+beta_right) #=grating period at phase match
#     minimum = min(abs(f-period)) #index of closest value
#     idx = np.where(abs(f-period) == minimum)
#     beta12Wav = Lambda.item(idx[0][0])
  
#     f= 2*math.pi/(2*beta_left)
#     minimum = min(abs(f-period))
#     idx = np.where(abs(f-period) == minimum)
#     beta1Wav = Lambda.item(idx[0][0])
  
#     f= 2*math.pi/(2*beta_right)
#     minimum = min(abs(f-period))
#     idx = np.where(abs(f-period) == minimum)
#     beta2Wav = Lambda.item(idx[0][0])
    
#     T =      np.zeros((1, Lambda.size),dtype=complex)
#     R =      np.zeros((1, Lambda.size),dtype=complex)
#     T_co =   np.zeros((1, Lambda.size),dtype=complex)
#     R_co =   np.zeros((1, Lambda.size),dtype=complex)
    
#     E_Thru = np.zeros((1, Lambda.size),dtype=complex)
#     E_Drop = np.zeros((1, Lambda.size),dtype=complex)
    
#     mode_kappa_a1=1
#     mode_kappa_a2=0 #no initial cross coupling
#     mode_kappa_b2=1
#     mode_kappa_b1=0
  
#     LeftRightTransferMatrix = np.zeros((4,4,Lambda.size),dtype=complex)
#     TopDownTransferMatrix = np.zeros((4,4,Lambda.size),dtype=complex)
#     InOutTransferMatrix = np.zeros((4,4,Lambda.size),dtype=complex)
  
#     # Apodization & segmenting
#     a = contraDC.apodization
#     l_seg = contraDC.N*period/N_seg
#     L_seg=l_seg
#     n_apodization=np.arange(N_seg)+0.5
#     zaxis= (np.arange(N_seg))*l_seg

#     if  a!=0:
#         kappa_apodG=np.exp(-a*((n_apodization)-0.5*N_seg)**2/N_seg**2)
#         ApoFunc=kappa_apodG

#     profile= (ApoFunc-min(ApoFunc))/(max(ApoFunc)-(min(ApoFunc))) # normalizes the profile

#     n_profile = np.linspace(0,N_seg,profile.size)
#     profile=np.interp(n_apodization, n_profile, profile)

#     if plot == True:
#         plt.figure(1)
#         plt.plot(zaxis*1e6, profile)
#         plt.ylabel('Profile (normalized)')
#         plt.xlabel('Length (um)')
        
#     kappaMin = contraDC.kappa_contra*profile[0]
#     kappaMax = contraDC.kappa_contra
    
#     kappa_apod=kappaMin+(kappaMax-kappaMin)*profile
  
#     lenghtLambda=Lambda.size
    
#     kappa_12max= max(kappa_apod)
    
#     couplingChirpFrac= np.zeros((1,N_seg))
#     lengthChirpFrac = np.zeros((1,N_seg))
#     chirpDev = 1 + couplingChirpFrac + lengthChirpFrac
    
#     n=np.arange(N_seg)
  
# #%% Beautiful Progress Bar
#     #https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
#     #Thank Greenstick.
    
#     # A List of Items
#     progressbar_width = lenghtLambda
#     # Initial call to print 0% progress
#     printProgressBar(0, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)
            
#     for ii in range(lenghtLambda):
#         #Update Bar
#         printProgressBar(ii + 1, progressbar_width, prefix = 'Progress:', suffix = 'Complete', length = 50)

#         randomChirp = np.zeros((1,N_seg))
#         chirpWL = chirpDev + randomChirp 
        
#         P=1
  
#         for n in range(N_seg):
#             L_0=(n)*l_seg

#             kappa_12=kappa_apod.item(n)
#             #kappa_21=conj(kappa_12); #unused: forward coupling!
#             kappa_11=contraDC.kappa_self1
#             kappa_22=contraDC.kappa_self2
      
#             beta_del_1=beta_left*chirpWL.item(n)-math.pi/period-j*alpha_e/2
#             beta_del_2=beta_right*chirpWL.item(n)-math.pi/period-j*alpha_e/2

#             # S1 = Matrix of propagation in each guide & direction
#             S_1=[  [j*beta_del_1.item(ii), 0, 0, 0], [0, j*beta_del_2.item(ii), 0, 0],
#                    [0, 0, -j*beta_del_1.item(ii), 0],[0, 0, 0, -j*beta_del_2.item(ii)]]

#             # S2 = transfert matrix
#             S_2=  [[-j*beta_del_1.item(ii),  0,  -j*kappa_11*np.exp(j*2*beta_del_1.item(ii)*L_0),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0)],
#                    [0,  -j*beta_del_2.item(ii),  -j*kappa_12*np.exp(j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  -j*kappa_22*np.exp(j*2*beta_del_2.item(ii)*L_0)],
#                    [j*np.conj(kappa_11)*np.exp(-j*2*beta_del_1.item(ii)*L_0),  j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  j*beta_del_1.item(ii),  0],
#                    [j*np.conj(kappa_12)*np.exp(-j*(beta_del_1.item(ii)+beta_del_2.item(ii))*L_0),  j*np.conj(kappa_22)*np.exp(-j*2*beta_del_2.item(ii)*L_0),  0,  j*beta_del_2.item(ii)]]

#             P0=np.matmul(scipy.linalg.expm(np.asarray(S_1)*l_seg),scipy.linalg.expm(np.asarray(S_2)*l_seg))
#             if n == 0:
#                 P1 = P0*P
#             else:
#                 P1 = np.matmul(P0,P)
                
#             P = P1
            
#         LeftRightTransferMatrix[:,:,ii] = P
        
#         # Calculating In Out Matrix
#         # Matrix Switch, flip inputs 1&2 with outputs 1&2
#         H = switchTop( P )
#         InOutTransferMatrix[:,:,ii] = H
        
#         # Calculate Top Down Matrix
#         P2 = P
#         # switch the order of the inputs/outputs
#         P2=np.vstack((P2[3][:], P2[1][:], P2[2][:], P2[0][:])) # switch rows
#         P2=swap_cols(P2,1,2) # switch columns
#         # Matrix Switch, flip inputs 1&2 with outputs 1&2
#         P3 = switchTop( P2 )
#         # switch the order of the inputs/outputs
#         P3=np.vstack((P3[3][:], P3[0][:], P3[2][:], P3[1][:])) # switch rows
#         P3=swap_cols(P3,2,3) # switch columns
#         P3=swap_cols(P3,1,2) # switch columns

#         TopDownTransferMatrix[:,:,ii] = P3
#         T[0,ii] = H[0,0]*mode_kappa_a1+H[0,1]*mode_kappa_a2
#         R[0,ii] = H[3,0]*mode_kappa_a1+H[3,1]*mode_kappa_a2

#         T_co[0,ii] = H[1,0]*mode_kappa_a1+H[1,0]*mode_kappa_a2
#         R_co[0,ii] = H[2,0]*mode_kappa_a1+H[2,1]*mode_kappa_a2

#         E_Thru[0,ii] = mode_kappa_a1*T[0,ii]+mode_kappa_a2*T_co[0,ii]
#         E_Drop[0,ii] = mode_kappa_b1*R_co[0,ii] + mode_kappa_b2*R[0,ii]

#         #%% return results
#         contraDC.E_Thru = E_Thru
#         contraDC.E_Drop = E_Drop
#         contraDC.wavelength = Lambda
#         contraDC.TransferMatrix = LeftRightTransferMatrix
        
#     return contraDC

# # Print iterations progress
# def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
#     """
#     Call in a loop to create terminal progress bar
#     @params:
#         iteration   - Required  : current iteration (Int)
#         total       - Required  : total iterations (Int)
#         prefix      - Optional  : prefix string (Str)
#         suffix      - Optional  : suffix string (Str)
#         decimals    - Optional  : positive number of decimals in percent complete (Int)
#         length      - Optional  : character length of bar (Int)
#         fill        - Optional  : bar fill character (Str)
#     """
#     percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
#     filledLength = int(length * iteration // total)
#     bar = fill * filledLength + '-' * (length - filledLength)
#     print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
#     # Print New Line on Complete
#     if iteration == total: 
#         print()