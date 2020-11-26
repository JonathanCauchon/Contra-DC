#%% append Python path to code location
import os,sys,inspect

# change directory for database
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir) 
os.chdir(parent_dir) 

# import ContraDC module
from ContraDC import *


def examples(num):
    """ Function implements 4 use-case examples """

    """ Example 1: regular SOI Contra-DC """
    if num ==1:

        # instantiate, simulate and show result
        device = ContraDC().simulate().displayResults()

        # calculate the group delay
        device.getGroupDelay()

        device = ContraDC().simulate().gen_sparams()

        # plot group delay
        plt.figure()
        plt.plot(device.wavelength*1e9, device.group_delay*1e12)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Tg (ps)")

        plt.show()



    """ Example 2: Full chirped example.
        Create a CDC with chirped w1, w2, period, temperature.
    """
    if num == 2:
    
        # Waveguide chirp
        w1 = [.56e-6, .562e-6]
        w2 = [.44e-6, .442e-6]
        w_chirp_step = .1e-9

        # Period chirp
        period = [318e-9, 322e-9]

        # apod shape
        apod_shape = "tanh"

        N = 1200

        device = ContraDC(N=N, w1=w1, w2=w2, apod_shape=apod_shape,
                         w_chirp_step=w_chirp_step, period=period)

        device.simulate().displayResults()



    """ Example 3: defining custom chirp profiles
    """
    if num == 3:

        device = ContraDC(apod_shape="tanh")

        z = np.linspace(0, device.N_seg, device.N_seg)
        device.w1_profile = device.w1*np.cos(z/600)
        device.w2_profile = device.w2*np.cos(z/600)

        device.simulate().displayResults()



    """ Example 4: using custom supermode indices.
        You might want to use this if you are designing 
        with silicon nitride, of using other waveguide specs than
        SOI, 100-nm gap.
    """
    if num == 4:

        device = ContraDC(polyfit_file="Tutorial/polyfit.txt", period=335e-9)
        device.simulate()

        plt.plot(device.wavelength*1e9, device.drop)
        plt.plot(device.wavelength*1e9, device.thru)
        plt.show()

    """Example 5: Lumerical-assisted flow
    """
    if num == 5:
        w1 = 560e-9
        w2 = 440e-9
        apod_shape = "tanh"
        N = 1200
        period = 318e-9

        device = ContraDC(N=N, w1=w1, w2=w2, apod_shape=apod_shape, period=period)
        device.simulate().displayResults()





if __name__ == "__main__":

    examples(5)

