from Modules import *
from ChirpedContraDC_v2 import *

N2 = [300, 350, 400]
Nin = [7, 8, 9]


# fig = plt.figure(figsize=(8, 6))
# grid = plt.GridSpec(6,3)

# ax1 = plt.axes()
# # ax1.subplot(grid[:,:])
# ax1.set_xlim([Nin[0]-.5, Nin[-1]+.5])
# ax1.set_xticks(Nin)
# ax1.set_xlabel("Number of Inner Gratings")

# ax1.set_yticks(N2)
# ax1.set_ylabel("N2")
# ax1.set_ylim([N2[0]-25, N2[-1]+25])

# for i in range(len(N2)):
# 	for j in range(len(Nin)):
# 		# print([N_in-.5, N_2-25, .5, 50])
# 		# ax2 = plt.axes([N_in-.5, N_2-25, 1, 50])
# 		ax2 = plt.axes([j/3.01, i/3.01, 1/5, 1/5])
# 		ax2.plot([1,2,4],[1,2,3])
i,j = np.meshgrid(N2, Nin)
fig, subfigs = plt.subplots(3 ,3, sharex="all", sharey="all", tight_layout=True, figsize=(9,6))
fig.figsize= (10,6)

for i, j, subfig in zip(i.reshape(-1),j.reshape(-1), subfigs.reshape(-1)):
	subfig.set_title("N2 = " + str(i)+", Nin = "+str(j))
	subfig.plot([1,2,3],[1,3,8])

plt.show()