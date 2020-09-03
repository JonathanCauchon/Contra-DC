"""
    MegaCDC class definition

    This is an attempt at creating an object that can simulate
    many ChirpedContraDC devices efficiently, in order to create
    datasets ^^. 

    Concept: based on ChirpedContraDC, but with an extra dimension
    dedicated for devices. Reaching deeper in dimensions.
"""

from utils import *
from modules import *
from ChirpedContraDC_v7 import *

class MegaCDC():

    def __init__(self, devices, wvl_range=[1500e-9,1600e-9], N_seg=101, resolution=1001,
                    alpha=10, N=1000):

        self.devices = devices
        self.wvl_range = wvl_range
        self.N_seg = N_seg
        self.resolution = resolution
        self.alpha = alpha
        self.N = N

        self._antiRefCoeff = 0.01

        if self.devices: # if device list is not empty
            self.apod_profile = np.stack([device.apod_profile for device in self.devices])
            self.period_profile = np.stack([device.period_profile for device in self.devices])
            self.beta1_profile = np.stack([device.beta1_profile for device in self.devices])
            self.beta2_profile = np.stack([device.beta2_profile for device in self.devices])


    def makeRightShape(self, param):
        param = np.expand_dims(param, (1))
        param = np.tile(param, (1,self.resolution,1))
        return param


    def simulate(self):

        alpha = self.alpha
        alpha_e = 100*self.alpha/10*np.log10(10)
        # alpha_e = self.makeRightShape(alpha_e)

        l_seg = self.N/self.N_seg * self.period_profile

        l_cum = np.cumsum(l_seg)
        l_cum -= l_cum[0]

        l_seg = self.makeRightShape(l_seg) 

        alpha_e = 100*self.alpha/10*np.log10(10)

        l_seg = self.N/self.N_seg * self.period_profile   
        l_cum = np.cumsum(l_seg, axis=-1)
        l_cum = l_cum - l_cum[:,0][0]

        # l_seg = self.makeRightShape(l_seg)
        l_cum = self.makeRightShape(l_cum)
        self.period_profile = self.makeRightShape(self.period_profile)

        kappa_12 = self.makeRightShape(self.apod_profile)
        kappa_21 = self.makeRightShape(np.conj(kappa_12))
        kappa_11 = self.makeRightShape(self._antiRefCoeff * self.apod_profile)
        kappa_22 = self.makeRightShape(self._antiRefCoeff * self.apod_profile)

        beta_del_1 = self.beta1_profile - np.pi/self.period_profile  - 1j*alpha_e
        beta_del_2 = self.beta2_profile - np.pi/self.period_profile  - 1j*alpha_e

        S1 = np.zeros((len(self.devices), self.resolution,self.N_seg,4,4), dtype=complex)
        S2 = np.zeros((len(self.devices), self.resolution,self.N_seg,4,4), dtype=complex)

        S1[:,:,:,0,0] = 1j*beta_del_1
        S1[:,:,:,1,1] = 1j*beta_del_2
        S1[:,:,:,2,2] = -1j*beta_del_1
        S1[:,:,:,3,3] = -1j*beta_del_2

        S2[:,:,:,0,0] = -1j*beta_del_1
        S2[:,:,:,1,0] = 0
        S2[:,:,:,2,0] = 1j*np.conj(kappa_11)*np.exp(-1j*2*beta_del_1*l_cum)
        S2[:,:,:,3,0] = 1j*np.conj(kappa_12)*np.exp(-1j*(beta_del_1+beta_del_2)*l_cum)

        S2[:,:,:,0,1] = 0
        S2[:,:,:,1,1] = -1j*beta_del_2
        S2[:,:,:,2,1] = 1j*np.conj(kappa_12)*np.exp(-1j*(beta_del_1+beta_del_2)*l_cum)
        S2[:,:,:,3,1] = 1j*np.conj(kappa_22)*np.exp(-1j*2*beta_del_2*l_cum)

        S2[:,:,:,0,2] = -1j*kappa_11*np.exp(1j*2*beta_del_1*l_cum)
        S2[:,:,:,1,2] = -1j*kappa_12*np.exp(1j*(beta_del_1+beta_del_2)*l_cum)
        S2[:,:,:,2,2] = 1j*beta_del_1
        S2[:,:,:,3,2] = 0

        S2[:,:,:,0,3] = -1j*kappa_12*np.exp(1j*(beta_del_1+beta_del_2)*l_cum)
        S2[:,:,:,1,3] = -1j*kappa_22*np.exp(1j*2*beta_del_2*l_cum)
        S2[:,:,:,2,3] = 0
        S2[:,:,:,3,3] = 1j*beta_del_2

        l_seg = np.expand_dims(l_seg, (1,3,4))
        l_seg = np.tile(l_seg, (1,self.resolution,1,4,4))

        M = expm(l_seg*(S1 + S2))

        # propagate the sucker
        for n in range(self.N_seg):
            P = M[:,:,n,:,:] if n == 0 else np.matmul(M[:,:,n,:,:],P)
            # P = np.clip(P, 0, None)

        left_right = P # left-right transfer matrix
        in_out = switchTop(left_right) # in-out transfer matrix

        self.E_thru = in_out[:,:,0,0]
        self.E_drop = in_out[:,:,3,0]

        # return results        
        self.thru = 10*np.log10(np.abs(self.E_thru)**2).squeeze()
        self.drop = 10*np.log10(np.abs(self.E_drop)**2).squeeze()

        # eliminate tiling
        self.period_profile = self.period_profile[:,0,:]

        # get group delay
        drop_phase = np.unwrap(np.angle(self.E_drop))
        frequency = self.devices[0].c/self.devices[0].wavelength
        omega = 2*np.pi*frequency

        group_delay = -np.diff(drop_phase)/np.diff(omega)
        self.group_delay = np.hstack((group_delay, np.expand_dims(group_delay[:,-1],1)))

        return self
 
        

class StandardDevice(ChirpedContraDC):
    """ The as-designed CDC """

    def __init__(self):
        # prop. constants
        wavelength = np.linspace(1500e-9, 1600e-9, 1001)
        n1_1550 = 2.6
        n2_1550 = 2.4
        dn1 = -1.0e6
        dn2 = -1.1e6
        centralWL = 1550e-9

        n1 = dn1*(wavelength - centralWL) + n1_1550
        n2 = dn2*(wavelength - centralWL) + n2_1550

        beta1 = 2*np.pi/wavelength * n1
        beta2 = 2*np.pi/wavelength * n2

        # instantiate CDC object w/ standard design
        super().__init__(resolution = 1001,
                          wvl_range = [1500e-9, 1600e-9],
                              N_seg = 101,
                              alpha = 0,
                              kappa = 54_500, 
                                  a = 10,
                                  N = 1_000)

        self.beta1_profile = np.transpose(np.tile(beta1, (101,1)))
        self.beta2_profile = np.transpose(np.tile(beta2, (101,1)))
        self.period_profile = 310e-9*np.ones(101)
        self.getApodProfile()

        self.basic_apod_profile = copy.copy(self.apod_profile)
        self.apod_distortion = None
        self.basic_period_profile = copy.copy(self.period_profile)
        self.period_distortion = None


    def distort(self):
        """ Apply distortions to apod & chirp profiles """

        # apod parameter changes
        self.a = np.random.uniform(.001, 15)
        self.getApodProfile()

        # kappa_max varies
        self.apod_distortion = np.random.normal(0.9, .1)
        self.apod_profile *= self.apod_distortion

        # random chirp is applied
        self.period_distortion = self.make_random_chirp()
        self.period_profile += self.period_distortion

        return self


    def make_random_chirp(self):
        z = np.arange(0,self.N_seg)
        x = np.array([0, np.random.randint(20,30), np.random.randint(45,55), np.random.randint(70,80), 100])
        y = np.random.normal(0, .2/3, size=x.shape[0])
        p = np.polyfit(x,y,3)

        return 1e-9*(p[0]*z**3 + p[1]*z**2 + p[2]*z + p[3])


num_devices = 1_000
import time

t0 = time.time()
devices = []
for i in range(num_devices):
    devices.append(StandardDevice().distort())

dataset = MegaCDC(devices, resolution=1001, N_seg=101, N=1000)

dataset.simulate()
print((time.time()-t0)/num_devices)

# fig, ax = plt.subplots(2,2)
# for i in range(len(dataset.devices)):
# ax[0,0].plot(np.linspace(1500,1600,1001), np.transpose(dataset.drop))
# ax[1,0].plot(np.linspace(1500,1600,1001), np.transpose(dataset.group_delay))
# ax[0,1].plot(np.arange(0,101), np.transpose(dataset.apod_profile))
# ax[1,1].plot(np.arange(0,101), np.transpose(dataset.period_profile))
# plt.show()

data = np.hstack((dataset.drop, 
                dataset.group_delay, 
                np.expand_dims(np.tile(dataset.N, num_devices),1), 
                dataset.apod_profile,
                dataset.period_profile))

if data.shape == (num_devices, 2205):
    np.save("MegaCDC_1k.npy", data)

# print(data.shape)
# plt.plot(data[0,:1001])
# plt.figure()
# plt.plot(data[0,1001:2002])
# print(data[0,2002], data[-1,2002])
# plt.figure()
# plt.plot(data[0,2003:-101])
# plt.figure()
# plt.plot(data[0,-101:])
# plt.show()






