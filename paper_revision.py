from modules import *
from ContraDC import *




# z = np.arange(0, 100)
# alpha = np.linspace(1,3,3)
# beta = np.linspace(2,4,3)

# for a in alpha:
#     for b in beta:
#         apod = 1/2 * (1 + np.tanh(b*(1-2*abs(2*z/100)**a)))
#         apod = np.append(np.flip(apod[0:int(apod.size/2)]), apod[0:int(apod.size/2)])
#         apod *= 48000

#         plt.plot(z, apod*1e-3, label="alpha = " + str(a) + ", beta = " + str(b))

# plt.legend()
# plt.xlabel("z (segments)")
# plt.ylabel("kappa (1/mm)")
# plt.show()

d = ContraDC(apod_shape="tanh", N=1000, N_seg=100)
d.T_profile = np.repeat(np.linspace(300,400,10),10)
d.simulate().displayResults()