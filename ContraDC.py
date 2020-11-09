""" 
    Contra-directional coupler model.

    Chirp your CDC, engineer your response.
    
    Based on Matlab model by Jonathan St-Yves
    as well as Python model by Mustafa Hammood.

    Created by Jonathan Cauchon, September 2019

    Last updated November 2020

"""


from modules import *
from utils import *


class ContraDC():
    """

    Contra-directional coupler class constructor. Defines parameters for simulation purposes.


    :param N: Number of grating periods.
    :type N: int

    :param period: Period of the grating [m]. If a float is passed,
        the period is considered uniform. If a list of 2 is passed,
        the period will be considered as a linear chirp from the first 
        to the second value given, with a step given by period_chirp_step.
    :type period: float or list

    :param polyfit_file: Path to the text file containing the polyfit for the supermode
        indices from a MODE simulation. The text file should follow the structured as:
        fit start wavelength, fit stop wavelength, w1 coefficient 1, w1 coefficient 2, 
        w2 coefficient 1, w2 coefficient 2. For instance: 1.5e-06,1.6e-06,1.97004,-201040,1.98997,-257755.
        If polyfit_file is None, the supermode indices will be interpolated for a 100-nm gap
        MODE simulation on the SOI platform, using w1 and w2. If not None, then the parameters
        w1, w2 and wvl_range have no impact on the simulation.
    :type polyfit_file: str, default=None

    :param resolution: Number of wavelength points to be used in the simulation.
    :type resolution: int

    :param N_seg: Number of grating segments to be used for propagation.
    :type N_seg: int

    :param wvl_range: Wavelength range to be used for simulation [m]. List of 2 elements where
        the simulations will be performed and plotted from first to second value.
    :type wvl_range: list

    :param alpha: Propagation loss in the grating [dB/cm].
    :type alpha: float

    :param w1: Width of waveguide 1 [m], if polyfit_file is None. If w1 is a float, w1 will
        be considered uniform. If w1 is a list of 2, w1 will be considered as linearly chirped 
        between the first and second value, following the chirp step given by w_chirp_step.
    :type w1: float, list

    :param w2: Width of waveguide 1 [m], if polyfit_file is None. If w2 is a float, w2 will
        be considered uniform. If w2 is a list of 2, w2 will be considered as linearly chirped 
        between the first and second value, following the chirp step given by w_chirp_step.
    :type w2: float, list

    :param apod_shape: Specifies the apodization profile shape, either "gaussian" or "tanh'.
    :type apod_shape: str

    :param a: Sepcifies the gaussian constant to be used in the apodization profile,
        if apod_shape is "gaussian".
    :type a: int

    :param kappa: Maximum coupling power [1/m].
    :type kappa: float

    :param T: Device Temperature [K]. If a float is passed, T is considered uniform.
        If a list of 2 is passed, the temperature is considered as linear along the 
        device, varying from the first value to the second value.
    :type T: float or list

    :param period_chirp_step: Chirp step of the period [m].
    :type period_chirp_step: float

    :param w_chirp_step: Chirp step of the waveguide widths [m].
    :type w_chirp_step: float

    :return: ContraDC object, not yet simulated.

    **Class Attributes**: Are calculated by the different member functions during simulation.
        They can be obverriden for custom contra-DC designs.

        - **apod_profile** (*np array*) -  Apodization profile, calculated by the getApodProfile() function.
        - **period_profile** (*np array*) - Period chirp profile along grating, calculated by getChirpProfile() fucntion.
        - **w1_profile** (*np array*) -  w1 chirp profile along grating, calculated by getChirpProfile() function.
        - **w2_profile** (*np array*) - w2 chirp profile along grating, calculated by getChirpProfile() function.
        - **T_profile** (*np array*) - Temperature chirp profile along grating, calculated by getChirpProfile() function.
        - **is_simulated** (*bool*) - *True* if simulation has taken place by invoking simulate(). 

    """



    def __init__(self, N=1000, period=322e-9, polyfit_file=None, a=10, apod_shape="gaussian",
        kappa=48000, T=300, resolution=500, N_seg=100, wvl_range=[1530e-9,1580e-9],
        central_wvl=1550e-9, alpha=10, w1=.56e-6, w2=.44e-6,
        w_chirp_step=1e-9, period_chirp_step=2e-9):

        # Class attributes
        self.N           =  N           
        self.period      =  period 
        self.polyfit_file = polyfit_file     
        self.a           =  a           
        self.kappa       =  kappa         
        self.T           =  T           
        self.resolution  =  resolution  
        self.N_seg       =  N_seg       
        self.alpha       =  alpha
        self.wvl_range   =  wvl_range   
        self.w1          =  w1          
        self.w2          =  w2          
        self.apod_shape  =  apod_shape  

        self.period_chirp_step = period_chirp_step # To comply with GDS resolution
        self.w_chirp_step = w_chirp_step

        # Constants
        self._antiRefCoeff = 0.01
        

        # Properties that will be set through methods
        self.apod_profile = None
        self.period_profile = None
        self.w1_profile = None
        self.w2_profile = None
        self.T_profile = None

        # Useful flag
        self.is_simulated = False



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
    


    def getPropConstants(self):
        """Calculates propagation constants,
        either through interpolation (for silicon), or through regression,
        given a text file containing the polyfit parameters (for nitride).

        :return: ContraDC object with calculated propagation constant profiles
            (self.beta1_profile, self.beta2_profile).
        """

        T0 = 300
        dneffdT = 1.87E-04      #[/K] assuming dneff/dn=1 (well confined mode)
        if self.T_profile is None:
            self.T_profile = self.T*np.ones(self.N_seg)

        neffThermal = dneffdT*(self.T_profile - T0)

        if self.polyfit_file is None:
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

        else:
            # polyfit of the type n1 = a1*wvl + b1, n2 = a2*wvl + b2
            with open("polyfit.txt", "r") as f:
                text = f.read()
                wvl1, wvl2, b1, a1, b2, a2 = text.split(",")


                self.wvl_range = [float(wvl1), float(wvl2)]
                wavelength_tiled = np.transpose(np.tile(self.wavelength, (self.N_seg,1)))

                self.n1_profile = neffThermal + float(a1)*wavelength_tiled + float(b1)
                self.n2_profile = neffThermal + float(a2)*wavelength_tiled + float(b2)
                self.beta1_profile = 2*math.pi / wavelength_tiled * self.n1_profile
                self.beta2_profile = 2*math.pi / wavelength_tiled * self.n2_profile

        return self         


    def getApodProfile(self):
        """Calculates the apodization profile, based on the apod_profile 
            (either "gaussian" of "tanh").

        :return: ContraDC object with calculated apodization profile (self.apod_profile).
        """

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
        """ Creates linear chirp profiles along the CDC device.
        Chirp is specified by assigning 2-element lists to the constructor
        for period, w1, w2 and T. The profiles are then created as linear, 
        and granularity is brought by the chirp_resolution specicfications 
        to match the fabrication process capabilities for realism (for instance, 
        w_chirp_step is set to 1 nm because GDS has a grid resolution of 1 nm for
        submission at ANT and AMF).

        :return: ContraDC object with calculated chirp profiles (self.period_profile, 
            self.w1_profile, self.w2_profile, self.T_profile).
        """

        if self.polyfit_file is None:

            

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

        # period chirp
        if self.period_profile is None:
            if isinstance(self.period, float):
                self.period = [self.period] # convert to list
            self.period_profile = np.linspace(self.period[0], self.period[-1], self.N_seg)
            # valid_periods = np.arange(self.period[0], self.period[-1] + self.period_chirp_step/100, self.period_chirp_step)

            # self.period_profile = np.repeat(valid_periods, round(self.N_seg/np.size(valid_periods)))
            # while np.size(self.period_profile) < self.N_seg:
            #     self.period_profile = np.append(self.period_profile, valid_periods[-1])
            # self.period_profile = np.round(self.period_profile, 15)
            # self.period_profile = self.period_profile[:self.N_seg+1]

        # temperature chirp
        if self.T_profile is None:
            if isinstance(self.T, float) or isinstance(self.T, int):
                self.T = [self.T] # convert to list
            self.T_profile = np.linspace(self.T[0],self.T[-1],self.N_seg)


        return self


    def makeRightShape(self, param):
        """ Simply adds dimensionality to the parameters in sights of 
        matrix operations in the "propagate" method The correct shape is
        (self.resolution, self.N_seg ,4, 4).

        :param param: Any ContraDC parameter.

        :return: The given parameter, with the right dimensionality.
        """

        param = np.expand_dims(param, (0))
        param = np.tile(param, (self.resolution,1))

        return param


    def propagate(self):
        """Propagates the optical field through the contra-DC, using the transfer-matrix
            method in a computationally-efficient way to calculate the total transfer 
            matrix and extract the thru and drop electric field responses.

        :return: ContraDC object with computed values for self.thru, self.drop, self.E_thru,
            self.E_drop, self.transferMatrix.
        """

        mode_kappa_a1, mode_kappa_b2 = 1, 1
        mode_kappa_a2, mode_kappa_b1 = 0, 0

        alpha_e = 100*self.alpha/10*np.log10(10)
        alpha_e = self.makeRightShape(alpha_e)

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


        # M contains EVERYTHING
        M = expm(l_seg*(S1 + S2))

        # propagate the sucker
        for n in range(self.N_seg):
            P = M[:,n,:,:] if n == 0 else np.matmul(M[:,n,:,:],P)

        # left-right transfer matrix
        left_right = P 

        # in-out transfer matrix
        in_out = switchTop(left_right) 

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
        """Calculates the group delay of the device,
        using the phase derivative. Requires self.is_simulated=True.
        """

        if self.is_simulated:
            drop_phase = np.unwrap(np.angle(self.E_drop))
            frequency = self.c/self.wavelength
            omega = 2*np.pi*frequency

            group_delay = -np.diff(drop_phase)/np.diff(omega)
            self.group_delay = np.append(group_delay, group_delay[-1])

            return self


    def simulate(self):
        """Simulates the contra-DC, in logical order as prescribed by the TMM method.
        Succintly calls self.getApodProfile(), self.getChirpProfile(), self.getPropConstants(),
        and self.propagate().

        :return: Simulated ContraDC object. 
        """

        self.getApodProfile()
        self.getChirpProfile()
        self.getPropConstants()
        self.propagate()

        return self
        

    def getPerformance(self):
        """ Calculates a couple of basic performance figures of the contra-DC,
        such as center wavelength, bandwidth, maximum reflection, etc.

        :return: ContraDC object, with a self.performance attibute containing the 
            performance data (self.performance).

        """

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



    def displayResults(self, tag_url=False):
        """Displays the result of the simulation in a user-friendly way.
        Convenient for design and optimization. Interface show the device's
        specifications and grating profiles, a graph of the spectral response, 
        as well as key performance figures calculated in getPerormance().

        :param tag_url: Either to tag the github repo URL or not.
        :type tag_url: bool

        """

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

        plt.subplot(grid[2,0])
        plt.plot(range(self.N_seg), self._period_profile)
        plt.xticks(profile_ticks, size=0)
        plt.yticks(color=text_color)
        plt.ylabel("$\Lambda$ (nm)", color=text_color)
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)

        

        if self.polyfit_file is None:
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

            plt.subplot(grid[3,0])
            plt.plot(range(self.N_seg), self.T_profile)
            plt.xticks(profile_ticks, size=0)
            plt.yticks(color=text_color)
            plt.ylabel("T (K)", color=text_color)
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

        return self

    def plot_format(self):

        plt.style.use('ggplot')
        plt.rcParams['axes.prop_cycle'] = cycler('color', ['blue', 'red', 'black', 'green', 'brown', 'orangered', 'purple'])
        plt.grid(b=True, color='w', linestyle='-', linewidth=1.5)
        plt.tick_params(axis=u'both', which=u'both',length=0)
        plt.legend(frameon=False)
