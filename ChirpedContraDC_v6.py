""" 
    Class ChripedContraDC_v6.py
    
    Chirped contra-directional coupler model
    Chirp your CDC, engineer your response
    
    Based on Matlab model by Jonathan St-Yves
    as well as Python model by Mustafa Hammood

    Jonathan Cauchon, Created September 2019
    Last updated June 2020

    -- v5 novelties --
    - Parallelization of TMM model: no more looping over wavelength
        (This is much faster)
    - Improved linalg operations to allow parallelization
    - Outsourced basic functions to utils.py

"""

"""   Notes
- Experimental vs simulated:
    Center wavelength: experimental is 8 nm higher than simulated center wvl

"""


from modules import *
from utils import *


class ChirpedContraDC():
    def __init__(self, N=1000, period=322e-9, a=10, apod_shape="gaussian",
        kappa=48000, T=300, resolution=500, N_seg=100, wvl_range=[1530e-9,1580e-9],
        central_wvl=1550e-9, alpha=10, stages=1, w1=.56e-6, w2=.44e-6,
        w_chirp_step=1e-9, period_chirp_step=2e-9):

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
        self.apod_shape  =  apod_shape  #  string described the shape of the coupling apodization []

        self.period_chirp_step = period_chirp_step # To comply with GDS resolution
        self.w_chirp_step = w_chirp_step

        # Note that gap is set to 100 nm

        # Constants
        self._antiRefCoeff = 0.01
        

        # Properties that will be set through methods
        self.apod_profile = None
        self.period_profile = None
        self.w1_profile = None
        self.w2_profile = None
        self.T_profile = None
        self.alpha_profile = None

        # Useful flag
        self.is_simulated = False

        
        # Dictionary conatining all units relative to the model
        self.units = {
                    "N"           :  None,   
                    "period"      :  "m",
                    "a"           :  None,       
                    "kappa"       :  "1/mm",   
                    "T"           :  "K",       
                    "resolution"  :  None,
                    "N_seg"       :  None,    
                    "alpha"       :  "dB/cm",    
                    "stages"      :  None,   
                    "wvl_range"   :  "m", 
                    "width"       :  "m",       
                    "target_wvl"  :  "m",
                    "apod_shape"  :  None,
                    "group delay" :  "s" }


    # return properties in user-friendly units
    @property
    def _wavelength(self):
        return self.wavelength*1e9

    @property
    def _period(self):
        return np.asarray(self.period)*1e9

    @property
    def _kappa(self):
        return self.kappa*1e-3

    @property
    def _apod_profile(self):
        return self.apod_profile*1e-3

    @property
    def _w1(self):
        return np.asarray(self.w1)*1e9  

    @property
    def _w2(self):
        return np.asarray(self.w2)*1e9

    @property
    def _period_profile(self):
        return self.period_profile*1e9

    @property
    def _w1_profile(self):
        return self.w1_profile*1e9  

    @property
    def _w2_profile(self):
        return self.w2_profile*1e9

    # Other non-changing properties
    @property
    def wavelength(self):
        return np.linspace(self.wvl_range[0], self.wvl_range[1], self.resolution)

    @property
    def c(self):
        return 299792458

    @property
    def l_seg(self):
        return self.N*np.mean(self.period)/self.N_seg

    @property
    def length(self):
        if self.period_profile is None:
            self.getChirpProfile()
        return np.round(np.sum(self.period_profile*self.N/self.N_seg), 9)
    


    # This performs a 3d interpolation to estimate effective indices
    def getPropConstants(self):
        
        T0 = 300
        dneffdT = 1.87E-04      #[/K] assuming dneff/dn=1 (well confined mode)
        if self.T_profile is None:
            self.T_profile = self.T*np.ones(self.N_seg)

        neffThermal = dneffdT*(self.T_profile - T0)

        # Import simulation results to be used for interpolation
        n1 = np.reshape(np.loadtxt("./Database/neff/neff_1.txt"),(5,5,5))
        n2 = np.reshape(np.loadtxt("./Database/neff/neff_2.txt"),(5,5,5))
        w1_w2_wvl = np.loadtxt("./Database/neff/w1_w2_lambda.txt")

        w1_tiled = np.tile(self.w1_profile, (self.resolution,1))
        w2_tiled = np.tile(self.w2_profile, (self.resolution,1))
        wavelength_tiled = np.transpose(np.tile(self.wavelength, (self.N_seg,1)))
        d = np.transpose(np.stack((w1_tiled, w2_tiled, wavelength_tiled)), (1,2,0))

        self.n1_profile = neffThermal + scipy.interpolate.interpn(w1_w2_wvl, n1, d)
        self.n2_profile = neffThermal + scipy.interpolate.interpn(w1_w2_wvl, n2, d)
        self.beta1_profile = 2*math.pi / wavelength_tiled * self.n1_profile
        self.beta2_profile = 2*math.pi / wavelength_tiled * self.n2_profile

        return self         


    def getApodProfile(self):
        if self.apod_profile is None:
            z = np.arange(0,self.N_seg)

            if self.apod_shape is "gaussian":
                if self.a == 0:
                    apod = self.kappa*np.ones(self.N_seg)
                else:
                    apod = np.exp(-self.a*(z - self.N_seg/2)**2 /self.N_seg**2)
                    apod = (apod - min(apod))/(max(apod) - min(apod))
                    apod *= self.kappa

            elif self.apod_shape is "tanh":
                z = np.arange(0, self.N_seg)
                alpha, beta = 2, 3
                apod = 1/2 * (1 + np.tanh(beta*(1-2*abs(2*z/self.N_seg)**alpha)))
                apod = np.append(np.flip(apod[0:int(apod.size/2)]), apod[0:int(apod.size/2)])
                apod *= self.kappa

            self.apod_profile = apod
            return self


    def getChirpProfile(self):

        # period chirp
        if self.period_profile is None:
            if isinstance(self.period, float):
                self.period = [self.period] # convert to list
            valid_periods = np.arange(self.period[0], self.period[-1] + self.period_chirp_step/100, self.period_chirp_step)

            self.period_profile = np.repeat(valid_periods, round(self.N_seg/np.size(valid_periods)))
            while np.size(self.period_profile) < self.N_seg:
                self.period_profile = np.append(self.period_profile, valid_periods[-1])
            self.period_profile = np.round(self.period_profile, 15)
            self.period_profile = self.period_profile[:self.N_seg+1]

        # Waveguide width chirp
        if self.w1_profile is None:
            if isinstance(self.w1, float):
                self.w1 = [self.w1] # convert to list
            self.w1_profile = np.linspace(self.w1[0],self.w1[-1],self.N_seg)
            self.w1_profile = np.round(self.w1_profile/self.w_chirp_step)*self.w_chirp_step
            self.w1_profile = np.round(self.w1_profile, 15)

        if self.w2_profile is None:
            if isinstance(self.w2, float):
                self.w2 = [self.w2] # convert to list
            self.w2_profile = np.linspace(self.w2[0],self.w2[-1],self.N_seg)
            self.w2_profile = np.round(self.w2_profile/self.w_chirp_step)*self.w_chirp_step
            self.w2_profile = np.round(self.w2_profile, 15)


        if self.T_profile is None:
            if isinstance(self.T, float) or isinstance(self.T, int):
                self.T = [self.T] # convert to list
            self.T_profile = np.linspace(self.T[0],self.T[-1],self.N_seg)

        if self.alpha_profile is None:
            self.alpha_profile = np.linspace(self.alpha, self.alpha, self.N_seg)

        return self


    def makeRightShape(self, param):
        param = np.expand_dims(param, (0))
        param = np.tile(param, (self.resolution,1))
        return param


    def propagate(self):

        mode_kappa_a1, mode_kappa_b2 = 1, 1
        mode_kappa_a2, mode_kappa_b1 = 0, 0

        alpha = self.makeRightShape(self.alpha_profile)
        alpha_e = 100*self.alpha_profile/10*np.log10(10)
        # alpha_e = self.makeRightShape(alpha_e)

        l_seg = self.N/self.N_seg * self.period_profile   
        l_cum = np.cumsum(l_seg)
        l_cum -= l_cum[0]

        l_seg = self.makeRightShape(l_seg)
        l_cum = self.makeRightShape(l_cum)

        kappa_12 = self.makeRightShape(self.apod_profile)
        kappa_21 = self.makeRightShape(np.conj(kappa_12))
        kappa_11 = self.makeRightShape(self._antiRefCoeff * self.apod_profile)
        kappa_22 = self.makeRightShape(self._antiRefCoeff * self.apod_profile)

        beta_del_1 = self.beta1_profile - np.pi/self.period_profile  - 1j*alpha_e
        beta_del_2 = self.beta2_profile - np.pi/self.period_profile  - 1j*alpha_e

        S1 = np.zeros((self.resolution,self.N_seg,4,4), dtype=complex)
        S2 = np.zeros((self.resolution,self.N_seg,4,4), dtype=complex)

        S1[:,:,0,0] = 1j*beta_del_1
        S1[:,:,1,1] = 1j*beta_del_2
        S1[:,:,2,2] = -1j*beta_del_1
        S1[:,:,3,3] = -1j*beta_del_2

        S2[:,:,0,0] = -1j*beta_del_1
        S2[:,:,1,0] = 0
        S2[:,:,2,0] = 1j*np.conj(kappa_11)*np.exp(-1j*2*beta_del_1*l_cum)
        S2[:,:,3,0] = 1j*np.conj(kappa_12)*np.exp(-1j*(beta_del_1+beta_del_2)*l_cum)

        S2[:,:,0,1] = 0
        S2[:,:,1,1] = -1j*beta_del_2
        S2[:,:,2,1] = 1j*np.conj(kappa_12)*np.exp(-1j*(beta_del_1+beta_del_2)*l_cum)
        S2[:,:,3,1] = 1j*np.conj(kappa_22)*np.exp(-1j*2*beta_del_2*l_cum)

        S2[:,:,0,2] = -1j*kappa_11*np.exp(1j*2*beta_del_1*l_cum)
        S2[:,:,1,2] = -1j*kappa_12*np.exp(1j*(beta_del_1+beta_del_2)*l_cum)
        S2[:,:,2,2] = 1j*beta_del_1
        S2[:,:,3,2] = 0

        S2[:,:,0,3] = -1j*kappa_12*np.exp(1j*(beta_del_1+beta_del_2)*l_cum)
        S2[:,:,1,3] = -1j*kappa_22*np.exp(1j*2*beta_del_2*l_cum)
        S2[:,:,2,3] = 0
        S2[:,:,3,3] = 1j*beta_del_2

        l_seg = np.expand_dims(l_seg, (2,3))
        l_seg = np.tile(l_seg, (1,1,4,4))

        # M1 = expm(S1*l_seg)
        # M2 = expm(S2*l_seg)

        # M contains EVERYTHING
        # M = np.matmul(M1,M2) 
        M = expm(l_seg*(S1 + S2))

        # propagate the sucker
        for n in range(self.N_seg):
            P = M[:,n,:,:] if n == 0 else np.matmul(M[:,n,:,:],P)
            # P = np.clip(P, 0, None)

        left_right = P # left-right transfer matrix
        in_out = switchTop(left_right) # in-out transfer matrix

        T = in_out[:,0,0]*mode_kappa_a1 + in_out[:,0,1]*mode_kappa_a2
        R = in_out[:,3,0]*mode_kappa_a1 + in_out[:,3,1]*mode_kappa_a2

        T_co = in_out[:,1,0]*mode_kappa_a1 + in_out[:,1,0]*mode_kappa_a2
        R_co = in_out[:,2,0]*mode_kappa_a1 + in_out[:,2,1]*mode_kappa_a2

        self.E_thru = mode_kappa_a1*T + mode_kappa_a2*T_co
        self.E_drop = mode_kappa_b1*R_co + mode_kappa_b2*R

        # return results        
        self.thru = 10*np.log10(np.abs(self.E_thru)**2).squeeze()
        self.drop = 10*np.log10(np.abs(self.E_drop)**2).squeeze()
        self.TransferMatrix = left_right
        self.is_simulated = True

        return self


    def getGroupDelay(self):
        if self.is_simulated:
            drop_phase = np.unwrap(np.angle(self.E_drop))
            frequency = self.c/self.wavelength
            omega = 2*np.pi*frequency

            group_delay = -np.diff(drop_phase)/np.diff(omega)
            self.group_delay = np.append(group_delay, group_delay[-1])

            return self


    def simulate(self):

        self.getApodProfile()
        self.getChirpProfile()
        self.getPropConstants()
        self.propagate()

        return self
        

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

            self.performance = {
                            "Ref. wvl" : [np.round(ref_wvl*1e9, 2), "nm"],
                            "BW"       : [np.round(bandwidth*1e9, 2), "nm"],
                            "Max ref." : [np.round(dropMax,2), "dB"],
                            "Avg ref." : [np.round(avg,2), "dB"],
                            "Std dev." : [np.round(std,2), "dB"] }


    # Display Plots and figures of merit 
    def displayResults(self, advanced=False, tag_url=False):

        self.getPerformance()

        fig = plt.figure(figsize=(9,6))

        plt.style.use('ggplot')
        plt.rcParams['axes.prop_cycle'] = cycler('color', ['blue', 'red', 'black', 'green', 'brown', 'orangered', 'purple'])

        profile_ticks = np.round(np.linspace(0, self.N_seg, 4))
        text_color = np.asarray([0,0,0]) + .25

        grid = plt.GridSpec(6,3)

        plt.subplot(grid[0:2,0])
        plt.title("Grating Profiles", color=text_color)
        plt.plot(range(self.N_seg), self._apod_profile)
        plt.xticks(profile_ticks, size=0)
        plt.yticks(color=text_color)
        plt.ylabel("$\kappa$ (/mm)", color=text_color)
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)

        plt.subplot(grid[2:4,0])
        plt.plot(range(self.N_seg), self._period_profile)
        plt.xticks(profile_ticks, size=0)
        plt.yticks(color=text_color)
        plt.ylabel("$\Lambda$ (nm)", color=text_color)
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)

        plt.subplot(grid[4,0])
        plt.plot(range(self.N_seg), self._w2_profile, label="wg 2")
        plt.ylabel("w2 (nm)", color=text_color)
        plt.xticks(profile_ticks, size=0, color=text_color)
        plt.yticks(color=text_color)
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)

        plt.subplot(grid[5,0])
        plt.plot(range(self.N_seg), self._w1_profile, label="wg 1")
        plt.xlabel("Segment", color=text_color)
        plt.ylabel("w1 (nm)", color=text_color)
        plt.xticks(profile_ticks, color=text_color)
        plt.yticks(color = text_color)
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)

        plt.subplot(grid[0:2,1])
        plt.title("Specifications", color=text_color)
        numElems = 6
        plt.axis([0,1,-numElems+1,1])
        plt.text(0.5,-0,"N : " + str(self.N),fontsize=11,ha="center",va="bottom", color=text_color)
        plt.text(0.5,-1,"N_seg : " + str(self.N_seg),fontsize=11,ha="center",va="bottom", color=text_color)
        plt.text(0.5,-2,"a : " + str(self.a),fontsize=11,ha="center",va="bottom", color=text_color)
        plt.text(0.5,-3,"p: " + str(self._period)+" nm",fontsize=11,ha="center",va="bottom", color=text_color)
        plt.text(0.5,-4,"w1 : " + str(self._w1)+" nm",fontsize=11,ha="center",va="bottom", color=text_color)
        plt.text(0.5,-5,"w2 : " + str(self._w2)+" nm",fontsize=11,ha="center",va="bottom", color=text_color)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)


        plt.subplot(grid[0:2,2])
        plt.title("Performance", color=text_color)
        numElems = len(self.performance)
        plt.axis([0,1,-numElems+1,1])
        for i, item  in zip(range(len(self.performance)), self.performance):
            plt.text(0.5,-i, item +" : ", fontsize=11, ha="right", va="bottom", color=text_color)
            plt.text(0.5,-i, str(self.performance[item][0])+" "+self.performance[item][1], fontsize=11, ha="left", va="bottom", color=text_color)
        plt.xticks([])
        plt.yticks([])
        plt.box(False)

        
        plt.subplot(grid[2:,1:])
        plt.plot(self.wavelength*1e9, self.thru, label="Thru port")
        plt.plot(self.wavelength*1e9, self.drop, label="Drop port")
        plt.legend(loc="best", frameon=False)
        plt.xlabel("Wavelength (nm)", color=text_color)
        plt.ylabel("Response (dB)", color=text_color)
        plt.tick_params(axis='y', which='both', labelleft=False, labelright=True, \
                        direction="in", right=True, color=text_color)
        plt.yticks(color=text_color)
        plt.xticks(color=text_color)
        # plt.tick_params(axis='x', top=True)
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)

        if tag_url:
            url = "https://github.com/JonathanCauchon/Contra-DC"
            plt.text(self._wavelength.min(), min(self.drop.min(), self.thru.min()), url, va="top", color="grey", size=6)

        plt.show()

    def plot_format(self):
        plt.style.use('ggplot')
        plt.rcParams['axes.prop_cycle'] = cycler('color', ['blue', 'red', 'black', 'green', 'brown', 'orangered', 'purple'])
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)
        plt.legend(frameon=False)


    # export design for easy GDS implementation
    def getGdsInfo(self, corrugations=[38e-9, 32e-9], gap=100e-9, plot=False):
        if self.apod_profile is None:
            self.getApodProfile()
        N_per_seg = int(self.N/self.N_seg)
        kappa = np.repeat(self.apod_profile, 2*N_per_seg)
        corru1 = kappa/self.kappa * corrugations[0]
        corru2 = kappa/self.kappa * corrugations[-1]

        self.getChirpProfile()

        w1 = np.repeat(self.w1_profile, 2*N_per_seg)
        w2 = np.repeat(self.w2_profile, 2*N_per_seg)
        w = np.hstack((w1, w2))
        

        self.getChirpProfile()

        # print(self.period_profile.shape)
        half_p = np.repeat(self.period_profile/2, 2*N_per_seg)
        # gds_w1 = self.w1*np.ones(2*self.N)
        # gds_w2 = self.w2*np.ones(2*self.N)

        z = np.cumsum(half_p)
        z -= z[0]
        # z = np.hstack((z, z))
        half_p = np.hstack((half_p, half_p))

        x1 = corru1/2*np.ones(2*self.N)
        x1[1::2] *= -1

        x2 = -w1/2 - gap - w2/2 + corru2/2*np.ones(2*self.N)
        x2[1::2] -= 2*corru2[1::2]/2
        x2 *= -1 # symmetric grating

        info_pos = np.hstack((np.vstack((z, x1)), np.vstack((z, x2)))).transpose()
        
        info_pos *= 1e6
        half_p *= 1e6

        if plot:
            plt.plot(info_pos[0:2*self.N,0], info_pos[0:2*self.N,1])
            plt.plot(info_pos[2*self.N:,0], info_pos[2*self.N:,1])
            plt.title("Rectangle centers")

            plt.figure()
            plt.plot(info_pos[:,0], w*1e6, ".")
            plt.title("WG Widths")

            plt.figure()
            plt.plot(info_pos[:,0], half_p, ".")
            plt.title("Half Period Profile")

            plt.show()

        self.gds_pos = info_pos
        self.gds_half_p = half_p
        self.gds_w = w*1e6

        return self.gds_pos, self.gds_half_p

    def exportGdsInfo(self, fileName="auto", plot=False): 
        self.getGdsInfo(plot=plot)
        data = np.vstack((self.gds_pos[:,0], self.gds_pos[:,1], self.gds_w, self.gds_half_p)).transpose()
        data = np.round(data, 3)

        if fileName == "auto":
            fileName = str(self.apod_shape)+"_N_"+str(self.N)+"_p_"+str(round(self.period_profile[0]*1e9))+"_"+str(round(self.period_profile[-1]*1e9))+"_Nseg_"+str(self.N_seg)
        
        np.savetxt("Designs/"+fileName+".txt", data, fmt="%4.3f")


    def __add__(cdc1, cdc2):
        if isinstance(cdc2, ChirpedContraDC):

            cdc3 = copy.copy(cdc1)

            if cdc1.apod_profile is None:
                cdc1.getApodProfile()
            if cdc2.apod_profile is None:
                cdc2.getApodProfile()
            if cdc1.period_profile is None:
                cdc1.getChirpProfile()
            if cdc2.period_profile is None:
                cdc2.getChirpProfile()

            cdc3.apod_profile = np.append(cdc1.apod_profile, cdc2.apod_profile)
            cdc3.period_profile = np.append(cdc1.period_profile, cdc2.period_profile)
            cdc3.w1_profile = np.append(cdc1.w1_profile, cdc2.w1_profile)
            cdc3.w2_profile = np.append(cdc1.w2_profile, cdc2.w2_profile)
            cdc3.alpha_profile = np.append(cdc1.alpha_profile, cdc2.alpha_profile)

            cdc3.N += cdc2.N
            cdc3.N_seg += cdc2.N_seg
            # cdc3.l_seg = np.append(cdc1.l_seg, cdc2.l_seg)
            # cdc3.z_seg = np.append(cdc1.z_seg, cdc2.z_seg+cdc1.z_seg[-1]+(cdc2.z_seg[1]-cdc2.z_seg[0]))

            # cdc1.getGdsInfo()
            # cdc2.getGdsInfo()

            # cdc3.gds_K = np.append(cdc1.gds_K, cdc2.gds_K)
            # cdc3.gds_z = np.append(cdc1.gds_z, cdc2.gds_z)
            # cdc3.gds_p = np.append(cdc1.gds_p, cdc2.gds_p)
            # cdc3.gds_w1 = np.append(cdc1.gds_w1, cdc2.gds_w1)
            # cdc3.gds_w2 = np.append(cdc1.gds_w2, cdc2.gds_w2)

        return cdc3












